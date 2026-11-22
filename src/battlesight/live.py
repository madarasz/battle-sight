"""Game core of the live app. No window, no camera, no GUI toolkit.

A front end reads frames from the camera and gives each one to
`LiveSession.step`. It shows the returned image and state, and it sends the
user's commands with `LiveSession.dispatch`.

The contract between the core and a front end (docs/specs/ui.md, section 8):

- `ACTIONS` lists all things that the user can do.
- `LiveSession.state()` is a JSON-safe dict with all that the user can see.

The image holds the board overlays only. Status text, messages and the name
menu belong to the front end.
"""

from __future__ import annotations

import math
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import cv2
import numpy as np

from . import ident as idt
from . import lightfit
from . import measure
from . import moves
from . import score as sc
from . import shoot
from . import state as savestate
from . import terrain as ter
from . import walls
from .calib import DriftMonitor, DriftReport, DriftStatus, TagDetector, board_corners_px, lens_for_frame
from .camera import CameraSettings
from .config import BoardConfig
from .colors import MEASURE_COLOR, MINIMAP_BACK_COLOR, MINIMAP_BOARD_COLOR, MOVE_MELEE_COLOR, NAME_SURE_COLOR, NO_TEAM_COLOR, POINTER_CROSS_COLOR, POINTER_CROSS_EDGE_COLOR, RANGE_ZONE_COLOR, SELECTED_COLOR, STATUS_ERROR_COLOR, STATUS_HINT_COLOR, STATUS_OK_COLOR, TEAM_COLORS
from .overlay import draw_base_discs, draw_bases, draw_board_corners, draw_circle_mm, draw_dashed_circle_mm, draw_dead_marks, draw_dashed_outline_mm, draw_grid, draw_health_bars, draw_idents, draw_melee_sign, draw_objective_arrows, draw_move_mm, draw_outline_mm, draw_path_mm, draw_range_mm, draw_rim_line, draw_tags, draw_targeting_mm, draw_terrain, fit_to_width
from .pipeline import BASES_MARGIN_MM, Pipeline, base_crops, bases_config, bases_image, find_base_near
from .rectify import fit_view, mm_to_rect, rect_transform, warp_board

SNAPSHOT_DIR = Path(__file__).resolve().parents[2] / "data" / "snapshots"
DEFAULT_REFERENCE = SNAPSHOT_DIR / "empty_board.png"
SAVES_DIR = Path(__file__).resolve().parents[2] / "data" / "saves"  # saved game states; the auto-save folders are here too
NO_AUTOSAVE = frozenset({"board.hover", "state.save"})  # every other action that reaches `dispatch` saves the state
DISPLAY_WIDTH = 1600  # image width in the windows; the windows can be resized
PREVIEW_PAD_MM = 16.0  # board around the base in the crop of the selected base
AIM_DIAMETER_MM = 40.0  # "Locate operative": board under the pointer as if a base of this size was there
OPENING_KINDS = ("hatchway", "breach")
OPENING_HIT_MM = 8.0  # a click this near to a hatchway or a breach point selects it
TEAM_SLOTS = 2  # one team list each side of the board
OPENING_NEAR_IN = 2  # bases this near a selected hatchway or breach point get a yellow rim


# --- helpers shared with the other live commands --------------------------------


class Notice:
    """Last message of a live window. `say` prints it (errors to stderr) and
    keeps it on the screen for `seconds`, so the user sees it without the
    Terminal. `clock` is for tests."""

    def __init__(self, seconds: float = 6.0, clock: Callable[[], float] = time.monotonic):
        self.seconds = seconds
        self.clock = clock
        self.text: str | None = None
        self.color = STATUS_HINT_COLOR
        self.level = "hint"
        self.until = 0.0

    def say(self, text: str, color=STATUS_HINT_COLOR, error: bool = False) -> None:
        print(text, file=sys.stderr if error else sys.stdout)
        self.text, self.color, self.until = text, color, self.clock() + self.seconds
        self.level = "error" if error else "done" if color == STATUS_OK_COLOR else "hint"

    def error(self, text: str) -> None:
        self.say(text, STATUS_ERROR_COLOR, error=True)

    def ok(self, text: str) -> None:
        self.say(text, STATUS_OK_COLOR)

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


class BoardTracker:
    """Board fit of live frames, shared by the live app and the enroll window.
    Finds the corner tags of each frame and follows them with a
    `DriftMonitor`. A still board uses the saved position, so the view does
    not jitter. The view also stays when a player covers one or two cards
    (`DriftReport.held`). No window calls."""

    def __init__(self, cfg: BoardConfig, camera: CameraSettings, frame_size: tuple[int, int] | None = None):
        self.cfg = cfg
        self.camera = camera
        self.detector = TagDetector(cfg)
        self.monitor = DriftMonitor(cfg)
        self._lens_size = frame_size
        self.lens = None if frame_size is None else lens_for_frame(camera, frame_size)
        self.obs: dict = {}
        self.report: DriftReport | None = None
        self.view = None  # rectify.BoardView of the last frame, None without a board fit

    def update(self, frame: np.ndarray) -> DriftReport:
        """Find the tags of `frame` and fit the board view. Returns the drift report."""
        size = (frame.shape[1], frame.shape[0])
        if self.lens is None or self._lens_size != size:
            self.lens, self._lens_size = lens_for_frame(self.camera, size), size
        self.obs = self.detector.detect(frame)
        self.report = report = self.monitor.update(self.obs)
        tags = self.monitor.reference if report.ok or report.held else self.obs
        self.view = None
        if tags and all(i in tags for i in self.cfg.tag_ids):
            try:
                self.view = fit_view(self.cfg, tags, self.lens)
            except ValueError as exc:
                print(f"rectify: {exc}", file=sys.stderr)
        return report


class BoardWindow:
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
        if self.view is None:
            return None
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


def load_reference(pipe: Pipeline, path: Path) -> np.ndarray | None:
    """Set the reference from the file `path`. Returns its image, or None when there is none."""
    if path.is_file():
        ref = cv2.imread(str(path), cv2.IMREAD_COLOR)
        try:
            pipe.set_reference(ref)
            print(f"empty-board reference: {path}")
            return ref
        except ValueError as exc:
            print(f"{path}: {exc}. Push e on the empty board to take a new one.", file=sys.stderr)
    return None


# --- the contract: actions ------------------------------------------------------


def _always(_state: dict) -> bool:
    return True


def _has_selection(state: dict) -> bool:
    return state["selected"] is not None


def _has_any_selection(state: dict) -> bool:
    return state["selected"] is not None or state["selected_opening"] is not None or state["selected_operative"] is not None or state["selected_light"] is not None


def _light_located(state: dict) -> bool:
    return state["selected_light"] is not None and state["selected_light"]["located"]


def _can_locate_selected_light(state: dict) -> bool:
    """"Identify terrain here": a selected card symbol and a piece file to search."""
    return state["selected_light"] is not None and not state["selected_light"]["located"] and not state["busy"] and bool(state["light_files"])


def _in_snapshot(state: dict) -> bool:
    return state["mode"] == "snapshot"


def shown_view(state: dict) -> str:
    """What the video shows: "live", "snapshot" or "minimap". The minimap
    covers the live video and the snapshot."""
    return "minimap" if state["minimap"] else state["mode"]


def _can_name(state: dict) -> bool:
    """The actions that name operatives need the identification model."""
    return not state["busy"] and state["model_ready"]


def _can_add(state: dict) -> bool:
    """A missed base can be added at any time: without a snapshot, the
    action takes one with no bases. Its name needs the identification model."""
    return _can_name(state) and not state["adding"]


def _can_locate_light(state: dict) -> bool:
    """With a piece file of the terrain type, also without a layout."""
    return not state["busy"] and not state["locating_light"] and bool(state["light_files"])


def _has_wounds(state: dict) -> bool:
    """The selected base has a name with maximum wounds."""
    w = state["selected_wounds"]
    return w is not None and w["max"] is not None


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
    radio: bool = False  # one of a set of radio buttons: `checked` is true for one of the set


# One text for each long action: the front end shows it with the progress bar (state key `progress`).
TASK_TEXT = {
    "snapshot.name": "Identifying operatives",
    "snapshot.bases": "Finding bases",
    "base.add": "Locating operative",
    "light.locate": "Identifying terrain",
    "light.locate_at": "Locating terrain",
}
PRELOAD_TEXT = "Loading models"

ACTIONS: tuple[Action, ...] = (
    Action("snapshot.name", "Identify all operatives", "space", enabled=_can_name),
    Action("snapshot.bases", "Find bases only", "b", enabled=lambda s: not s["busy"], control=False),
    Action("snapshot.back", "Go back to Live", "Esc", control=False),
    Action("view.top_down", "Adjust perspective", "v", checked=lambda s: s["top_down"]),
    Action("view.grid", "Grid", "g", checked=lambda s: s["grid"]),
    Action("view.terrain", "Layout", checked=lambda s: s["terrain"], enabled=lambda s: s["map_layout"] is not None),
    Action("view.zones", "Drop zones", checked=lambda s: s["zones"], enabled=lambda s: s["map_layout"] is not None),
    Action("view.keys", "Terrain keys", checked=lambda s: s["keys"], enabled=lambda s: s["map_layout"] is not None),
    Action("view.light", "Light terrain", checked=lambda s: s["light_shown"], enabled=lambda s: s["map_layout"] is not None or bool(s["light"])),
    Action("view.live", "Live", checked=lambda s: shown_view(s) == "live", radio=True),
    Action("view.snapshot", "Snapshot", checked=lambda s: shown_view(s) == "snapshot", enabled=lambda s: s["snapshot_ok"] is not None or s["resolution"] is not None, radio=True),
    Action("view.minimap", "Minimap", checked=lambda s: shown_view(s) == "minimap", enabled=lambda s: s["map_layout"] is not None, radio=True),
    Action("view.marks", "Control marks", checked=lambda s: s["marks"], enabled=lambda s: any(b["label"] is not None for b in s["bases"])),
    Action("measure.walls", "Measure around walls", "w", checked=lambda s: s["around_walls"], enabled=lambda s: s["map_layout"] is not None),
    Action("map.terrain_type", "Terrain type", args=("name",)),
    Action("map.layout", "Layout", args=("name",)),
    Action("team.set", "Team", args=("slot", "name")),
    Action("reference.take", "Take empty-board reference", "e"),
    Action("reference.clear", "Clear reference", "c", enabled=lambda s: s["reference_set"]),
    Action("measure.move", "Measure movement", "m", checked=lambda s: s["movement"]),
    Action("measure.avoid_melee", "Avoid control range", checked=lambda s: s["avoid_melee"], enabled=lambda s: s["movement"]),
    Action("measure.shoot", "Shoot at", checked=lambda s: s["shooting"], enabled=_has_selection),
    Action("board.click", "Click the board", args=("x", "y"), enabled=_in_snapshot),
    Action("board.hover", "Point at the board", args=("x", "y")),
    Action("board.select", "Right click the board", args=("x", "y"), enabled=_in_snapshot),
    Action("base.choose", "Set name", args=("label", "index"), enabled=_has_selection),
    Action("base.select", "Select operative", args=("index",)),
    Action("operative.select", "Select operative of a team", args=("label",)),
    Action("roster.add", "Add operative", args=("slot", "label")),
    Action("roster.remove", "Remove operative", args=("slot",)),
    Action("base.deselect", "Clear selection", enabled=_has_any_selection),
    Action("base.remove", "Remove ID", enabled=_has_selection),
    Action("base.add", "Locate operative", enabled=_can_add),
    *(Action(f"ring.{n}", f"Ring {n} in", str(n), enabled=_has_selection, checked=partial(lambda n, s: s["ring_in"] == n, n)) for n in range(1, 10)),
    Action("ring.clear", "Clear ring", "0", enabled=lambda s: s["ring_in"] is not None),
    Action("wounds.minus", "Wound -1", "-", enabled=_has_wounds),
    Action("wounds.plus", "Wound +1", "+", enabled=_has_wounds),
    Action("wounds.set", "Set wounds", args=("value",), enabled=_has_wounds),
    Action("score.skull", "Score skull", args=("slot", "category", "tp", "skull")),
    Action("score.tac_op", "Tac Op", args=("slot", "name")),
    Action("score.primary", "Primary", args=("slot", "value")),
    Action("score.cp", "Command points", args=("slot", "delta")),
    Action("game.turn", "Turning point", args=("delta",)),
    Action("game.crit_op", "Crit Op", args=("name",)),
    Action("opening.toggle", "Open / close", "o", enabled=lambda s: s["selected_opening"] is not None),
    Action("light.locate", "Identify all terrain", enabled=lambda s: s["light_can_locate"]),
    Action("light.locate_at", "Locate terrain", enabled=_can_locate_light),
    Action("light.remove", "Remove ID", enabled=_light_located),
    Action("light.locate_here", "Identify terrain here", enabled=_can_locate_selected_light),
    Action("names.accept", "Accept all identification", "a", enabled=lambda s: s["unconfirmed"] > 0),
    Action("names.undo", "Undo last gallery photo", "u", enabled=lambda s: s["can_undo"]),
    Action("frame.save", "Save screenshot", "s"),
    Action("state.save", "Save state", "k"),
    Action("view.layout", "Front end layout", args=("name",)),
    Action("app.layout", "Next layout", "l", shell=True),
    Action("app.help", "Keys", "h", shell=True),
    Action("app.quit", "Quit", "q", shell=True),
)


# --- the core -------------------------------------------------------------------


@dataclass
class LiveFrame:
    """Result of one `LiveSession.step`. `image` is None until the first camera
    frame. `scale` is image pixels per source pixel. `crop` is the board around
    the selected base or the selected identified light terrain piece, or None."""

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
        pieces_root: Path | str = ter.DEFAULT_PIECES_ROOT,
        saves_dir: Path | str | None = None,
        autosave: bool = True,
    ):
        self.cfg = cfg
        self.board = BoardTracker(cfg, camera, frame_size)
        self.lens = self.board.lens
        self.pipe = Pipeline(cfg, None, camera)
        self.reference_path = None if reference_path is None else Path(reference_path)  # None: no load, no save
        self.snapshot_dir = Path(snapshot_dir)
        self.reference_frame: np.ndarray | None = None  # the empty-board photo of the reference; a saved state keeps it
        if self.reference_path is not None:
            self.reference_frame = load_reference(self.pipe, self.reference_path)
        self.saves_dir = Path(SAVES_DIR if saves_dir is None else saves_dir)
        self.autosave_enabled = autosave
        self.autosave_path = self.saves_dir / time.strftime("autosave-%Y%m%d-%H%M%S")  # new at each start, written all run; the folder appears at the first save
        self._save_cache: dict[str, Any] = {}
        self._save_problem: str | None = None
        self.ui_layout: str | None = None  # name of the front end layout; the front end sets it, a saved state keeps it
        self.detector = self.board.detector
        self.monitor = self.board.monitor
        self.win = BoardWindow(cfg, display_width)
        self.clock = clock
        self.notice = Notice(clock=clock)
        self.teams = list(teams) if teams else None
        self.gallery_root = gallery_root
        self.team_choices = idt.list_teams(gallery_root)
        slots = list(self.teams or [])[:TEAM_SLOTS]
        self.team_slots: list[str | None] = slots + [None] * (TEAM_SLOTS - len(slots))
        self._team_ops: dict[str, list[idt.Operative]] = {}  # operatives of each team, read once
        self._team_infos: dict[str, idt.TeamInfo] = {}  # team values of each team file, read once
        # Team list of each side: labels (copy labels for an operative with `max` > 1)
        # in list order; None = one copy of each operative of the team file.
        self.in_play: list[list[str] | None] = [None] * TEAM_SLOTS
        # Last known board centre (mm) of each copy of an operative with `max` > 1:
        # identification tells the copies apart by it.
        self.positions: dict[str, np.ndarray] = {}
        self.gallery: idt.Gallery | None = None
        self._update_roster()
        self.gallery_tried = False

        self.top_down = True
        self.grid = True
        self.terrain = True  # terrain overlay on, when a map layout is chosen
        self.zones = False  # drop zones of the layout off
        self.keys = False  # wall keys (A1, B2, ...) off
        self.light_shown = True  # light terrain (card symbols, found outlines) on
        self.marks = False  # melee signs and objective arrows of named operatives
        self.minimap = False  # a drawn board (layout, grid, bases) in place of the camera; needs a layout
        self.around_walls = False  # measure around walls, not through them; on at each layout load
        self.movement = False  # the ring is the area that the selected base can move to
        self.avoid_melee = False  # the movement keeps 1 in away from the enemy bases
        self.shooting = False  # "Shoot at": targeting lines from the pointer to the selected base
        self._shot: tuple[Any, shoot.Shot | None, int | None] | None = None  # key, last shot, shooter base
        self.hover_mm: np.ndarray | None = None  # board position of the mouse pointer
        self._move_field: tuple[Any, moves.MoveField, list[np.ndarray]] | None = None  # key, field, outline
        self._move_route: tuple[Any, Any] | None = None  # key, route to the pointer
        self._navigator: tuple[Any, walls.Navigator] | None = None  # for the layout and its open openings
        self._wall_path: tuple[Any, Any] | None = None  # last path around walls, with its key
        self._wall_range: tuple[Any, Any] | None = None  # last range area, with its key
        self._ring_bases: tuple[Any, list[int]] | None = None  # bases in the ring distance, with its key
        self.maps_root = Path(maps_root)
        self.pieces_root = Path(pieces_root)  # piece files: real outlines of the light terrain
        self.terrain_map: ter.TerrainMap | None = None
        self.pieces: dict[str, ter.Piece] = {}
        self.light_poses: dict[int, tuple[tuple[float, float], float]] = {}  # light item -> (centre mm, angle) found on the board
        self.loose_light: list[tuple[str, tuple[float, float], float]] = []  # (key, centre mm, angle): pieces found by "Locate terrain" that are no item of the layout
        self.selected_light: int | None = None  # index in the light shapes: the items of `terrain_map.light`, then `loose_light`
        self.locating_light = False  # "Locate terrain": the next click on the board is on a light terrain piece
        self._light_version = 0  # goes up when a light terrain outline changes
        self._locate_image: np.ndarray | None = None  # "Locate terrain": the top-down image at its start (pointer crop)
        self._light_crop_cache: tuple[tuple | None, np.ndarray | None] = (None, None)  # (selection key, crop) of `_light_crop`
        self.maps: list[ter.MapEntry] = []
        self.terrain_type: str | None = None
        self.map_layout: str | None = None
        self.terrain_shapes: list[ter.Shape] = []
        self.openings: list[ter.Shape] = []  # hatchways and breach points of the layout
        self.selected_opening: int | None = None  # index in `openings`
        self.opened: set[int] = set()  # indexes of the open openings; a layout starts with all closed
        self._read_maps()
        self.busy = False
        self.camera_up = True
        self.on_busy: Callable[[LiveFrame], None] | None = None  # called before a long action
        self.on_progress: Callable[[dict[str, Any]], None] | None = None  # called at each step of a long action, with the `progress` state
        self._task_progress: dict[str, Any] | None = None  # the long action that runs (`_task`): text, done, total
        self._preload: dict[str, Any] | None = None  # progress of `start_preload` while it runs: done, total, error
        self._preload_thread: threading.Thread | None = None

        self.frame: np.ndarray | None = None
        self.obs: dict = {}
        self.report: DriftReport | None = None
        self.view = None  # rectify.BoardView of the last frame
        self.last_status = None
        self.image: np.ndarray | None = None
        self.detect_ms: float | None = None  # tag detection and board fit of the last frame
        self.meter = FpsMeter()

        self.result = None  # FrameResult of the last snapshot, or None
        self.frozen = False  # True: the snapshot is on the screen, False: the live video
        self.snap: np.ndarray | None = None  # camera frame of the snapshot
        self.session: idt.LabelSession | None = None
        self.selected: int | None = None  # index of the selected base
        self.selected_operative: str | None = None  # label chosen in a team list that has no base in the snapshot
        self.target: int | None = None  # index of the second base of a distance
        self.ring_in: int | None = None  # ring on the selected base, in inches
        self.adding = False  # "Locate operative": the next click on the board is the centre of a missed base
        self.new_base: int | None = None  # base that "Locate operative" made and the user did not name yet
        self.right_click: str | None = None  # what the last right click did: "name" (select for names) or "measure"
        self.shown: np.ndarray | None = None  # rendered snapshot
        self.dirty = True
        # Game state. Current wounds of each operative label that is not at its
        # maximum; the names of a new snapshot keep them.
        self.wounds: dict[str, int] = {}
        # Score of each side of the board (not of a team): it stays when the team changes.
        self.scores = [sc.Score() for _ in range(TEAM_SLOTS)]
        self.turn = sc.TURNS[0]  # turning point of the game, shared by both sides
        self.crit_op: str | None = None  # Crit Op of the game, shared by both sides

        self._handlers: dict[str, Callable[..., None]] = {
            "snapshot.name": lambda: self._snapshot(name=True),
            "snapshot.bases": lambda: self._snapshot(name=False),
            "snapshot.back": self._back,
            "view.top_down": self._toggle_view,
            "view.grid": self._toggle_grid,
            "view.terrain": self._toggle_terrain,
            "view.zones": self._toggle_zones,
            "view.keys": self._toggle_keys,
            "view.light": self._toggle_light,
            "view.live": self._show_live,
            "view.snapshot": self._show_snapshot,
            "view.minimap": self._show_minimap,
            "view.marks": self._toggle_marks,
            "measure.walls": self._toggle_around_walls,
            "measure.move": self._toggle_movement,
            "measure.avoid_melee": self._toggle_avoid_melee,
            "measure.shoot": self._toggle_shooting,
            "board.hover": self._hover,
            "map.terrain_type": self._set_terrain_type,
            "map.layout": self._set_map_layout,
            "team.set": self._set_team,
            "reference.take": self._take_reference,
            "reference.clear": self._clear_reference,
            "board.click": self._click,
            "board.select": self._select_for_names,
            "base.choose": self._choose,
            "base.select": self._select_index,
            "operative.select": self._select_operative,
            "roster.add": self._roster_add,
            "roster.remove": self._roster_remove,
            "base.deselect": self._deselect,
            "base.remove": self._remove_base,
            "base.add": self._start_adding,
            **{f"ring.{n}": partial(self._set_ring, n) for n in range(1, 10)},
            "ring.clear": partial(self._set_ring, None),
            "wounds.minus": partial(self._add_wounds, -1),
            "wounds.plus": partial(self._add_wounds, 1),
            "wounds.set": self._set_wounds,
            "score.skull": self._score_skull,
            "score.tac_op": self._score_tac_op,
            "score.primary": self._score_primary,
            "score.cp": self._score_cp,
            "game.turn": self._game_turn,
            "game.crit_op": self._game_crit_op,
            "opening.toggle": self._toggle_opening,
            "light.locate": self._locate_light,
            "light.locate_at": self._start_locating_light,
            "light.remove": self._remove_light,
            "light.locate_here": self._locate_selected_light,
            "names.accept": self._accept_all,
            "names.undo": self._undo,
            "frame.save": self._save_frame,
            "state.save": self._save_state,
            "view.layout": self._set_ui_layout,
        }

    # --- model preload ----------------------------------------------------------

    def start_preload(self) -> None:
        """Load the identification model on a thread of its own, so that the
        first "Identify all operatives" does not wait for it. The state key
        `progress` shows the load steps. The actions that need the model are
        off until the end (`model_ready`). A model that is loaded already
        (also "color-hist") starts no thread."""
        params = idt.IdentParams.from_config(self.cfg)
        embedder = idt.make_embedder(params.model)
        if embedder.loaded or self._preload_thread is not None:
            return
        self._preload = {"done": 0, "total": 1, "error": None}

        def progress(done: int, total: int, _text: str) -> None:
            self._preload = {"done": done, "total": total, "error": None}  # one assignment: the worker thread reads it

        def run() -> None:
            started = time.perf_counter()
            try:
                embedder.load(params.input_px, progress)
            except Exception as exc:  # the first identification tries again
                traceback.print_exc()
                self._preload = {"done": 0, "total": 1, "error": str(exc)}
                return
            print(f"model {params.model} loaded in {time.perf_counter() - started:.1f} s")
            self._preload = None

        self._preload_thread = threading.Thread(target=run, name="model-preload", daemon=True)
        self._preload_thread.start()

    def _model_ready(self) -> bool:
        preload = self._preload
        return preload is None or preload["error"] is not None

    def _progress_state(self) -> dict[str, Any] | None:
        """The long action that runs (`_task`), else the model load while it
        runs: one text for the whole task, steps done and all steps."""
        if self._task_progress is not None:
            return dict(self._task_progress)
        preload = self._preload
        if preload is None or preload["error"] is not None:
            return None
        return {"text": PRELOAD_TEXT, "done": preload["done"], "total": preload["total"]}

    @contextmanager
    def _task(self, text: str) -> Iterator[Callable[[int, int], None]]:
        """A long action: `busy` while it runs, and `progress` with `text`.
        The front end gets a frame at the start (`on_busy`) and the progress
        at each step (`on_progress`). Yields `step(done, total)`."""
        self.busy = True
        self._task_progress = {"text": text, "done": 0, "total": 1}

        def step(done: int, total: int) -> None:
            self._task_progress = {"text": text, "done": min(done, total), "total": max(total, 1)}
            if self.on_progress is not None:
                self.on_progress(dict(self._task_progress))

        try:
            if self.on_busy is not None:
                self.on_busy(self.current())
            yield step
        finally:
            self.busy = False
            self._task_progress = None

    def _check_preload(self) -> None:
        """A failed preload becomes a notice once (on the thread of the session)."""
        preload = self._preload
        if preload is not None and preload["error"] is not None:
            self._preload = None
            self.notice.error(f"model not loaded: {preload['error']}")

    def _need_model(self) -> bool:
        if self._model_ready():
            return True
        self.notice.say("the identification model is loading: wait for the end")
        return False

    # --- frames -----------------------------------------------------------------

    def step(self, frame: np.ndarray | None) -> LiveFrame:
        """Process one camera frame. None = the camera is down."""
        self._check_preload()
        self.camera_up = frame is not None
        if frame is None:
            self.meter.reset()
            if self._minimap_on():  # the minimap needs no camera
                self.image = self._render_minimap()
            return self.current()
        self.frame = frame
        started = self.clock()
        self.meter.tick(started)
        report = self.board.update(frame)
        self.detect_ms = (self.clock() - started) * 1000.0
        self.obs, self.report, self.view, self.lens = self.board.obs, report, self.board.view, self.board.lens
        if report.relocked:
            self.notice.ok("board moved: new position saved")
        if report.status is not self.last_status:
            print(report.summary())
            self.last_status = report.status
        if self._minimap_on():
            self.image = self._render_minimap()
        else:
            self.image = self._snapshot_image() if self.frozen else self._render_live()
        return self.current()

    def current(self) -> LiveFrame:
        """The last image with the state of now."""
        state = self.state()
        return LiveFrame(self.image, state, self.win.scale, self._selected_crop())

    def _selected_crop(self) -> np.ndarray | None:
        """While "Locate operative" waits for a click or "Shoot at" is on: the
        board under the pointer (`_aim_crop`). Else the selected base with `PREVIEW_PAD_MM` of board around it, cut
        from the centre of its gallery crop (`ident.store_crop`). Else the
        selected identified light terrain piece (`_light_crop`)."""
        if self.adding or self.shooting:
            return self._aim_crop()
        if self.locating_light:
            return self._aim_crop(self._locate_image, self._piece_span_mm())
        if self.selected_light is not None:
            return self._light_crop()
        s, i = self.session, self.selected
        if s is None or i is None or i >= len(s.crops):
            return None
        crop, diameter = s.crops[i], float(s.diameters_mm[i])
        side = crop.shape[0]
        px_per_mm = side / (diameter + 2 * idt.STORE_PAD_MM)
        keep = min(side, int(round((diameter + 2 * PREVIEW_PAD_MM) * px_per_mm)))
        start = (side - keep) // 2
        return np.ascontiguousarray(crop[start : start + keep, start : start + keep])  # a QImage needs one block

    def _light_crop(self) -> np.ndarray | None:
        """The board image around the selected identified light terrain
        piece: its outline bounding box, square, with `PREVIEW_PAD_MM` each
        side. From the snapshot while it is on the screen, else from the
        frame at the time of the selection (one board warp per selection)."""
        k, outlines = self.selected_light, self._light_outlines()
        if k is None or k >= len(outlines) or not self._light_state()[k]["located"]:
            return None
        key = (k, self._light_version, self.frozen, id(self.result))
        if self._light_crop_cache[0] != key:
            image, crop = self._light_image(), None
            if image is not None:
                lo, hi = outlines[k].min(axis=0), outlines[k].max(axis=0)
                side = int(round((float(np.max(hi - lo)) + 2 * PREVIEW_PAD_MM) * self.cfg.bases_px_per_mm))
                cx, cy = mm_to_rect(bases_config(self.cfg), ((lo + hi) / 2)[None])[0]
                crop = np.ascontiguousarray(cv2.getRectSubPix(image, (side, side), (float(cx), float(cy))))
            self._light_crop_cache = (key, crop)
        return self._light_crop_cache[1]

    def _piece_span_mm(self) -> float:
        """"Locate terrain": the pointer crop holds the largest piece file
        outline around the pointer."""
        return max((2.0 * float(np.linalg.norm(p.outline, axis=1).max()) for p in self.pieces.values()), default=AIM_DIAMETER_MM)

    def _aim_crop(self, img: np.ndarray | None = None, span_mm: float = AIM_DIAMETER_MM) -> np.ndarray | None:
        """The board around the pointer, `span_mm` plus `PREVIEW_PAD_MM`
        each side (the size of the crop of a selected base of that size),
        with a cross at the pointer. `img`: a top-down bases image, else the
        snapshot one. None when the pointer is off the board image."""
        if img is None:
            img = None if self.result is None else self.result.scratch.get("bases_image")
        at = self.hover_mm
        if img is None or at is None:
            return None
        ppm = self.cfg.bases_px_per_mm
        cx, cy = mm_to_rect(bases_config(self.cfg), np.asarray(at, float)[None])[0]
        h, w = img.shape[:2]
        if not (0 <= cx < w and 0 <= cy < h):
            return None
        side = int(round((span_mm + 2 * PREVIEW_PAD_MM) * ppm))
        crop = cv2.getRectSubPix(img, (side, side), (float(cx), float(cy)))
        c, arm = side // 2, max(4, side // 12)
        for color, width in ((POINTER_CROSS_EDGE_COLOR, 3), (POINTER_CROSS_COLOR, 1)):
            cv2.line(crop, (c - arm, c), (c + arm, c), color, width)
            cv2.line(crop, (c, c - arm), (c, c + arm), color, width)
        return np.ascontiguousarray(crop)

    def set_display_width(self, width: int) -> None:
        """Largest width of the returned image. A front end sets it to the width on the screen."""
        self.win.display_width = width
        self.dirty = True

    def _render_live(self) -> np.ndarray:
        """The live video. The bases and names of the last snapshot stay on
        it, in board mm, until the next snapshot."""
        cfg, frame, view = self.cfg, self.frame, self.view
        if self.top_down and view is not None:
            out = warp_board(frame, view, cfg)
        else:
            out = frame.copy()
            draw_tags(out, self.obs, cfg, self.report)
            reference = self.monitor.reference
            draw_board_corners(out, board_corners_px(cfg, self.obs), board_corners_px(cfg, reference) if reference else None)
        if view is not None:
            self._draw_board(out)
        return self.win.fit(out)

    def _minimap_on(self) -> bool:
        return self.minimap and self.map_layout is not None

    def _render_minimap(self) -> np.ndarray:
        """A drawn top-down board in place of the camera image: the layout,
        the grid, a disc for each base of the last snapshot with its name and
        health bar, and the selection. Same pixels per mm as the top-down view."""
        cfg = self.cfg
        _, (w, h) = rect_transform(cfg)
        img = np.full((h, w, 3), MINIMAP_BACK_COLOR, np.uint8)
        (x0, y0), (x1, y1) = np.rint(mm_to_rect(cfg, np.array([[0.0, 0.0], [cfg.width_mm, cfg.height_mm]]))).astype(int)
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), MINIMAP_BOARD_COLOR, -1)
        self._draw_board(img, minimap=True)
        return self.win.fit(img)

    def _snapshot_image(self) -> np.ndarray:
        if self.dirty:
            self.shown = self._render_snapshot()
            self.dirty = False
        return self.shown

    def _render_snapshot(self) -> np.ndarray:
        r, win = self.result, self.win
        if not r.ok:
            return win.fit(self.pipe.annotate(self.snap, r, banner=False))
        img = win.base_image(self.snap, r)
        self._draw_board(img)
        return win.fit(img)

    def _draw_board(self, img: np.ndarray, minimap: bool = False) -> None:
        """Terrain, grid, the bases of the last snapshot and the selection,
        with the mapping of the shown image. `minimap`: the layout shows also
        with "Layout" off, and each base is a full disc."""
        to_px = self._to_px
        self._draw_terrain(img, to_px, always=minimap)
        if self.grid:
            draw_grid(img, self.cfg, to_px)
        if self.selected_light is not None and self.light_shown:
            draw_outline_mm(img, self._light_outlines()[self.selected_light], to_px, SELECTED_COLOR)
        if self.selected_opening is not None:
            points = self.openings[self.selected_opening].points
            draw_outline_mm(img, points, to_px, SELECTED_COLOR)
            draw_outline_mm(img, measure.offset_outline(points, measure.MM_PER_INCH), to_px, RANGE_ZONE_COLOR)
            draw_dashed_outline_mm(img, measure.offset_outline(points, OPENING_NEAR_IN * measure.MM_PER_INCH), to_px, RANGE_ZONE_COLOR)
        found = self._found()
        colors = {team: color for team, color in zip(self.team_slots, TEAM_COLORS) if team is not None}
        if minimap and found:
            labels = [x.label for x in self.session.idents] if self.session is not None else [None] * len(found)
            draw_base_discs(img, found, [NO_TEAM_COLOR if label is None else colors.get(label.partition("/")[0], NAME_SURE_COLOR) for label in labels], to_px)
        if self.session is not None:
            control = self._control(found) if self.marks else None
            if control:
                objectives = self._objectives()
                for i, k in control["objectives"]:
                    draw_objective_arrows(img, found[i].center_mm, found[i].diameter_mm, objectives[k], to_px)
            draw_idents(img, found, self.session.idents, to_px, scores=False, team_colors=colors, inside=minimap)
            bars = []
            for c, x in zip(found, self.session.idents):
                current, maximum = self._wounds_of(x.label)
                if maximum is not None:
                    bars.append((c.center_mm, c.diameter_mm, current, maximum))
            draw_health_bars(img, bars, to_px)
            draw_dead_marks(img, [(c.center_mm, c.diameter_mm) for c, x in zip(found, self.session.idents) if self._is_dead(x.label)], to_px)
            if control:
                for i, j in control["melee"]:
                    _, start, end = measure.rim_gap(found[i].center_mm, found[i].diameter_mm, found[j].center_mm, found[j].diameter_mm)
                    draw_melee_sign(img, 0.5 * (start + end), to_px)
        elif found:
            draw_bases(img, found, to_px)
        for i in self._near_opening(found):  # bases near the selected hatchway or breach point
            draw_circle_mm(img, found[i].center_mm, 0.5 * found[i].diameter_mm, to_px, RANGE_ZONE_COLOR)
        self._draw_measure(img, found)
        self._draw_shot(img, found)

    def _near_opening(self, found: list) -> list[int]:
        """Indexes of the bases with the rim within `OPENING_NEAR_IN` of the
        selected hatchway or breach point."""
        if self.selected_opening is None or not found:
            return []
        points = self.openings[self.selected_opening].points
        return measure.near_polygon([c.center_mm for c in found], [c.diameter_mm for c in found], points, OPENING_NEAR_IN * measure.MM_PER_INCH)

    def _draw_measure(self, img: np.ndarray, found: list) -> None:
        """The selected base, its ring, the second base and the distance."""
        if self.selected is None:
            return
        to_px, a = self._to_px, found[self.selected]
        draw_circle_mm(img, a.center_mm, 0.5 * a.diameter_mm, to_px, SELECTED_COLOR)
        around = self._walls_on()
        if self.movement:  # 1 in around each enemy: a move into it is a melee
            for i in self._enemy_indexes(found):
                draw_dashed_circle_mm(img, found[i].center_mm, 0.5 * found[i].diameter_mm + measure.MM_PER_INCH, to_px, RANGE_ZONE_COLOR)
        if self.ring_in is not None and self.movement:
            field, outline = self._field(found)
            draw_range_mm(img, outline, to_px, MEASURE_COLOR)
            route = self._route(found)
            if route is not None:
                draw_move_mm(img, route[1], field.radius, route[2], route[0], to_px)
                for i in self._move_melee(found):
                    draw_melee_sign(img, found[i].center_mm, to_px, MOVE_MELEE_COLOR)
        elif self.ring_in is not None and around:
            draw_range_mm(img, self._range_around(a), to_px, MEASURE_COLOR)
        elif self.ring_in is not None:
            draw_circle_mm(img, a.center_mm, measure.ring_radius_mm(a.diameter_mm, self.ring_in), to_px, MEASURE_COLOR)
        for i in self._in_ring(found):  # other bases in the ring distance
            draw_circle_mm(img, found[i].center_mm, 0.5 * found[i].diameter_mm, to_px, RANGE_ZONE_COLOR)
        if self.target is not None:
            b = found[self.target]
            draw_circle_mm(img, b.center_mm, 0.5 * b.diameter_mm, to_px, MEASURE_COLOR)
            if around:
                path = self._path_around(a, b)
                if path is not None:  # no line when the walls close all paths
                    draw_path_mm(img, path[1], f'{path[0] / measure.MM_PER_INCH:.2f}"', to_px)
            else:
                gap, start, end = measure.rim_gap(a.center_mm, a.diameter_mm, b.center_mm, b.diameter_mm)
                draw_rim_line(img, start, end, f'{gap / measure.MM_PER_INCH:.2f}"', to_px)

    # --- control marks --------------------------------------------------------------

    def _objectives(self) -> list[tuple[float, float]]:
        """Objective marker centres of the layout (none without a layout)."""
        return [s.center for s in self.terrain_shapes if s.kind == "objective"]

    def _control(self, found: list) -> dict[str, list[list[int]]] | None:
        """`melee`: pairs of base indexes of named operatives of two teams
        within 1 in (rim to rim); `objectives`: [base index, objective index]
        of named operatives within 1 in of an objective marker (needs a
        layout). None without names. Dead operatives are in no melee."""
        if self.session is None or not found:
            return None
        teams = [None if x.label is None else x.label.partition("/")[0] for x in self.session.idents]
        centers, diameters = [c.center_mm for c in found], [c.diameter_mm for c in found]
        alive = [None if self._is_dead(x.label) else t for x, t in zip(self.session.idents, teams)]
        named = [i for i, t in enumerate(teams) if t is not None]
        contacts = measure.objective_contacts([centers[i] for i in named], [diameters[i] for i in named], self._objectives(), self.cfg.terrain_objective_mm, self.cfg.terrain_objective_range_mm)
        return {
            "melee": [[i, j] for i, j in measure.melee_pairs(centers, diameters, alive)],
            "objectives": [[named[i], k] for i, k in contacts],
        }

    def _toggle_marks(self) -> None:
        self.marks = not self.marks
        self.dirty = True

    # --- movement -------------------------------------------------------------------

    def _enemy_indexes(self, found: list) -> list[int]:
        """Indexes of the bases of the other team than the selected base.
        Bases without a name are no enemies."""
        if self.session is None or self.selected is None:
            return []
        labels = [x.label for x in self.session.idents]
        mine = labels[self.selected]
        if mine is None:
            return []
        team = mine.partition("/")[0]
        return [i for i, label in enumerate(labels[: len(found)]) if label is not None and label.partition("/")[0] != team]

    def _enemies(self, found: list) -> list[tuple[tuple[float, float], float]]:
        """(centre, diameter) of the enemy bases: they block the movement of the selected base."""
        return [((float(found[i].center_mm[0]), float(found[i].center_mm[1])), float(found[i].diameter_mm)) for i in self._enemy_indexes(found)]

    def _move_melee(self, found: list) -> list[int]:
        """Indexes of the enemy bases that the move to the pointer comes
        within 1 in of (rim to rim), at any point of the move."""
        route = self._route(found)
        if route is None:
            return []
        a = found[self.selected]
        return [i for i in self._enemy_indexes(found) if measure.path_distance(route[1], found[i].center_mm) < 0.5 * a.diameter_mm + 0.5 * found[i].diameter_mm + measure.MM_PER_INCH - 0.5]

    def _toggle_avoid_melee(self) -> None:
        self.avoid_melee = not self.avoid_melee
        self.dirty = True

    def _field(self, found: list) -> tuple[moves.MoveField, list[np.ndarray]]:
        """Movement field of the selected base for the ring distance, and its outline."""
        a, enemies = found[self.selected], self._enemies(found)
        # The base can go through all other bases, but it cannot stop on one.
        friends = [(xy, d) for i, c in enumerate(found) if i != self.selected for xy, d in [((float(c.center_mm[0]), float(c.center_mm[1])), float(c.diameter_mm))] if (xy, d) not in enemies]
        avoid = measure.MM_PER_INCH if self.avoid_melee else 0.0
        key = (self.map_layout, frozenset(self.opened), self._base_key(a), self.ring_in, tuple(enemies), tuple(friends), avoid)
        if self._move_field is None or self._move_field[0] != key:
            size = (self.cfg.width_mm, self.cfg.height_mm)
            rects, gaps = (), ()
            if self.map_layout is not None:
                shapes = self._shapes_now()
                xs, ys = self.cfg.grid_lines_mm()
                rects, gaps = walls.obstacles(shapes, size, (xs[0], ys[0], xs[-1], ys[-1])), walls.open_gaps(shapes)
            field = moves.move_field(a.center_mm, a.diameter_mm, self.ring_in, rects, gaps, enemies, size, friends, avoid)
            self._move_field = (key, field, field.outline())
            self._move_route = None
        return self._move_field[1], self._move_field[2]

    def _route(self, found: list):
        """The increments from the selected base to the pointer, or None."""
        if not self.movement or self.selected is None or self.ring_in is None or self.hover_mm is None or not found:
            return None
        field, _ = self._field(found)
        key = (self._move_field[0], float(self.hover_mm[0]), float(self.hover_mm[1]))
        if self._move_route is None or self._move_route[0] != key:
            self._move_route = (key, field.route(self.hover_mm))
        return self._move_route[1]

    def _move_state(self) -> dict[str, Any] | None:
        """`in`: inches of the move to the pointer, `increments`: inches of each straight increment."""
        route = self._route(self._found())
        return None if route is None else {"in": int(route[0]), "increments": [int(c) for c in route[2]]}

    def _toggle_movement(self) -> None:
        self.movement = not self.movement
        self.dirty = True

    def _hover(self, x: float | None, y: float | None) -> None:
        """The mouse pointer is at this pixel of the image (None: not on the image)."""
        at = None if x is None or y is None or not self._mapped() else self._to_mm(x, y)
        if (at is None) != (self.hover_mm is None) or (at is not None and not np.allclose(at, self.hover_mm)):
            self.hover_mm = at
            if (self.movement and self.ring_in is not None) or self.shooting:
                self.dirty = True

    # --- shooting -------------------------------------------------------------------

    def _toggle_shooting(self) -> None:
        """"Shoot at": targeting lines from the pointer to the selected base.
        Again (or `Esc`): stop."""
        if not self.shooting and self.selected is None:
            self.notice.say("click a base first")
            return
        self.shooting = not self.shooting
        self.dirty = True

    def _shooter_index(self, found: list) -> int | None:
        """The base under the pointer when it is not the selected base and
        not of the same team (a base without a name can be the shooter)."""
        if self.hover_mm is None:
            return None
        i = measure.base_at([c.center_mm for c in found], [c.diameter_mm for c in found], self.hover_mm, reach_mm=0.0)
        if i is None or i == self.selected:
            return None
        labels = [x.label for x in self.session.idents] if self.session is not None else []
        mine = labels[self.selected] if self.selected < len(labels) else None
        theirs = labels[i] if i < len(labels) else None
        if mine is not None and theirs is not None and mine.partition("/")[0] == theirs.partition("/")[0]:
            return None
        return i

    def _aim(self, found: list) -> tuple[shoot.Shot | None, int | None]:
        """The targeting lines from the pointer to the selected base and the
        shooter base under the pointer (kept until the pointer, a base or the
        terrain changes). (None, None) when "Shoot at" is off."""
        if not self.shooting or self.selected is None or self.hover_mm is None or self.selected >= len(found):
            return None, None
        a = found[self.selected]
        shooter = self._shooter_index(found)
        key = (
            self.map_layout,
            self._light_version,
            frozenset(self.opened),
            self._base_key(a),
            float(self.hover_mm[0]),
            float(self.hover_mm[1]),
            None if shooter is None else self._base_key(found[shooter]),
        )
        if self._shot is None or self._shot[0] != key:
            base = None if shooter is None else (found[shooter].center_mm, found[shooter].diameter_mm)
            self._shot = (key, shoot.aim(self.hover_mm, a.center_mm, a.diameter_mm, self._heavy_rects(), base, light=self._light_outlines(cover=True)), shooter)
        return self._shot[1], self._shot[2]

    def _light_outlines(self, cover: bool = False) -> list[np.ndarray]:
        """Outlines of the light terrain, in board mm, in the order of
        `selected_light`. `cover`: only the ones that give cover (no covered
        card symbol)."""
        return [s.points for s in self.terrain_shapes if s.kind == "light" and not (cover and s.covered)]

    def _heavy_rects(self) -> np.ndarray:
        """Heavy terrain of the layout: walls, pillars, closed hatchways and breach points."""
        if self.map_layout is None:
            return np.zeros((0, 4))
        return walls.obstacles(self._shapes_now())

    def _draw_shot(self, img: np.ndarray, found: list) -> None:
        """1 in around the target and the shooter, the targeting lines, the
        intervening heavy terrain and the light terrain that gives cover."""
        if not self.shooting or self.selected is None or self.selected >= len(found):
            return
        to_px, a = self._to_px, found[self.selected]
        draw_dashed_circle_mm(img, a.center_mm, 0.5 * a.diameter_mm + measure.MM_PER_INCH, to_px, RANGE_ZONE_COLOR)
        shot, shooter = self._aim(found)
        if shooter is not None:
            b = found[shooter]
            draw_dashed_circle_mm(img, b.center_mm, 0.5 * b.diameter_mm + measure.MM_PER_INCH, to_px, RANGE_ZONE_COLOR)
        if shot is not None:
            outlines = self._light_outlines(cover=True)
            draw_targeting_mm(img, shoot.facing_outline(shot.origin, a.center_mm, a.diameter_mm), shot.pieces, to_px, light_mm=[outlines[k] for k in shot.light])

    def _shot_state(self) -> dict[str, Any] | None:
        """`obscured`: from the heavy terrain in the targeting lines;
        `in_cover`: from heavy or light terrain (`cover_by`: "heavy", "light",
        "both" or None); `too_near`: terrain is in the target zone, but the
        target is within 2 in of the shooter; `intervening`: heavy pieces;
        `shooter`: base index under the pointer. None while "Shoot at" is off
        or the pointer is off the board or on the target."""
        shot, shooter = self._aim(self._found())
        if shot is None:
            return None
        return {
            "obscured": shot.obscured,
            "in_cover": shot.in_cover,
            "cover_by": shot.cover_by,
            "too_near": shot.too_near,
            "intervening": len(shot.pieces),
            "shooter": shooter,
        }

    # --- measurement around walls ---------------------------------------------------

    def _walls_on(self) -> bool:
        return self.around_walls and self.map_layout is not None

    def _nav(self) -> tuple[Any, walls.Navigator]:
        """The navigator of the layout with its open openings, and its key."""
        key = (self.map_layout, frozenset(self.opened))
        if self._navigator is None or self._navigator[0] != key:
            size = (self.cfg.width_mm, self.cfg.height_mm)
            xs, ys = self.cfg.grid_lines_mm()
            rects = walls.obstacles(self._shapes_now(), size, (xs[0], ys[0], xs[-1], ys[-1]))
            self._navigator = (key, walls.Navigator(rects, size))
        return self._navigator

    @staticmethod
    def _base_key(c) -> tuple[float, float, float]:
        return (float(c.center_mm[0]), float(c.center_mm[1]), float(c.diameter_mm))

    def _path_around(self, a, b) -> tuple[float, np.ndarray] | None:
        """Shortest path between two base rims around the walls (kept until a base or the terrain changes)."""
        nav_key, nav = self._nav()
        key = (nav_key, self._base_key(a), self._base_key(b))
        if self._wall_path is None or self._wall_path[0] != key:
            self._wall_path = (key, nav.path(a.center_mm, a.diameter_mm, b.center_mm, b.diameter_mm))
        return self._wall_path[1]

    def _range_around(self, a) -> list[np.ndarray]:
        """Outline of the area within the ring distance of the base, around the walls."""
        nav_key, nav = self._nav()
        key = (nav_key, self._base_key(a), self.ring_in)
        if self._wall_range is None or self._wall_range[0] != key:
            self._wall_range = (key, nav.reach(a.center_mm, a.diameter_mm, self.ring_in * measure.MM_PER_INCH))
        return self._wall_range[1]

    def _in_ring(self, found: list) -> list[int]:
        """Indexes of the other bases with the rim within the ring distance of
        the selected base (around the walls when that is on). Empty while
        movement is measured."""
        if self.selected is None or not self.ring_in or self.movement or not found:
            return []
        a, reach = found[self.selected], self.ring_in * measure.MM_PER_INCH
        others = [i for i in range(len(found)) if i != self.selected]
        around = self._walls_on()
        nav_key, nav = self._nav() if around else (None, None)
        key = (nav_key, self.selected, self.ring_in, tuple(self._base_key(c) for c in found))
        if self._ring_bases is None or self._ring_bases[0] != key:
            inside = []
            for i in others:
                b = found[i]
                if around:
                    path = nav.path(a.center_mm, a.diameter_mm, b.center_mm, b.diameter_mm)
                    gap = np.inf if path is None else path[0]
                else:
                    gap = measure.rim_gap(a.center_mm, a.diameter_mm, b.center_mm, b.diameter_mm)[0]
                if gap <= reach:
                    inside.append(i)
            self._ring_bases = (key, inside)
        return self._ring_bases[1]

    def _toggle_around_walls(self) -> None:
        self.around_walls = not self.around_walls
        self.dirty = True

    # --- mapping of the shown image -------------------------------------------------

    def _shown_view(self) -> tuple[bool, Any]:
        """(top-down, BoardView) of the image on the screen. The view is None
        when the live frame has no board fit."""
        if self._minimap_on():
            return True, None
        if self.frozen:
            return self.win.top_down, self.win.view
        return self.top_down and self.view is not None, self.view

    def _mapped(self) -> bool:
        """The shown image has a board mapping."""
        top_down, view = self._shown_view()
        return top_down or view is not None

    def _to_px(self, pts_mm) -> np.ndarray:
        """Board mm -> pixels of the shown image, before `fit`."""
        top_down, view = self._shown_view()
        if top_down:
            return mm_to_rect(self.cfg, pts_mm)
        return view.mm_to_px(np.asarray(pts_mm, float).reshape(-1, 2))

    def _to_mm(self, x: float, y: float) -> np.ndarray | None:
        """Pixel of the returned image -> board mm, or None without a mapping."""
        if self.frozen and not self._minimap_on():
            return self.win.to_mm(x, y)
        top_down, view = self._shown_view()
        px = np.array([[x / self.win.scale, y / self.win.scale]])
        if top_down:
            return (px / self.cfg.rectify_px_per_mm - self.cfg.rectify_margin_mm)[0]
        return None if view is None else view.px_to_mm(px)[0]

    # --- state ------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """All that a front end can show. JSON-safe."""
        if self.frozen and not self._minimap_on():
            self._snapshot_image()  # brings the scale and the view up to date
        s = self.session
        current = self.notice.current()
        self._update_kill_scores()
        return {
            "mode": "snapshot" if self.frozen else "live",
            "busy": self.busy,
            "model_ready": self._model_ready(),
            "progress": self._progress_state(),
            "top_down": self.win.top_down if self.frozen else self.top_down,
            "grid": self.grid,
            "terrain": self.terrain,
            "zones": self.zones,
            "keys": self.keys,
            "light_shown": self.light_shown,
            "minimap": self._minimap_on(),
            "marks": self.marks,
            "control": self._control(self._found()) if self.marks else None,
            "around_walls": self.around_walls,
            "movement": self.movement,
            "avoid_melee": self.avoid_melee,
            "move_melee": self._move_melee(self._found()),
            "move": self._move_state(),
            "shooting": self.shooting,
            "shot": self._shot_state(),
            "terrain_types": sorted({e.terrain for e in self.maps}),
            "terrain_type": self.terrain_type,
            "map_layouts": [e.name for e in self.maps if e.terrain == self.terrain_type],
            "map_layout": self.map_layout,
            "drift": self._drift_state(),
            "camera": self.camera_up,
            "lens": self.lens.source,
            "resolution": None if self.frame is None else [int(self.frame.shape[1]), int(self.frame.shape[0])],  # newest frame
            "teams": sorted(self.gallery.team_dirs) if self.gallery is not None else self.teams,
            "team_choices": list(self.team_choices),
            "team_slots": list(self.team_slots),
            "rosters": self._rosters_state(),
            "scores": [x.to_dict(self._tac_ops_of(slot)) for slot, x in enumerate(self.scores)],
            "turn": self.turn,
            "turns": list(sc.TURNS),
            "crit_op": self.crit_op,
            "crit_ops": list(sc.CRIT_OPS),
            "reference_set": self.pipe.has_reference,
            "snapshot_ok": None if self.result is None else bool(self.result.ok),
            "naming": s is not None,
            "bases": self._bases_state(),
            "selected": self.selected,
            "selected_operative": self.selected_operative,
            "selected_label": self._selected_label(),
            "adding": self.adding,
            "new_base": self.new_base,
            "right_click": self.right_click,
            "selected_opening": None if self.selected_opening is None else {"index": self.selected_opening, "kind": self.openings[self.selected_opening].kind, "open": self.selected_opening in self.opened},
            "near_opening": self._near_opening(self._found()),
            "light": self._light_state(),
            "selected_light": None if self.selected_light is None else self._light_state()[self.selected_light],
            "locating_light": self.locating_light,
            "light_files": sorted(self.pieces),
            "light_can_locate": self.terrain_map is not None and any(item.key in self.pieces for item in self.terrain_map.light),
            "openings": [{"index": i, "kind": sh.kind, "open": i in self.opened} for i, sh in enumerate(self.openings)],
            "selected_wounds": self._selected_wounds_state(),
            "target": self.target,
            "ring_in": self.ring_in,
            "in_ring": self._in_ring(self._found()),
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

    def _distance_state(self) -> dict[str, Any] | None:
        """`around`: measured around the walls; then `mm` and `in` are None when there is no path."""
        if self.selected is None or self.target is None:
            return None
        found = self._found()
        a, b = found[self.selected], found[self.target]
        if self._walls_on():
            path = self._path_around(a, b)
            if path is None:
                return {"mm": None, "in": None, "around": True}
            return {"mm": _num(path[0]), "in": _num(path[0] / measure.MM_PER_INCH), "around": True}
        gap = measure.rim_gap(a.center_mm, a.diameter_mm, b.center_mm, b.diameter_mm)[0]
        return {"mm": _num(gap), "in": _num(gap / measure.MM_PER_INCH), "around": False}

    def _bases_state(self) -> list[dict[str, Any]]:
        found = self._found()
        if not found:
            return []
        centers = np.array([c.center_mm for c in found]).reshape(-1, 2)
        if self._mapped():
            positions = [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in np.asarray(self._to_px(centers), float).reshape(-1, 2) * self.win.scale]
        else:  # live frame without a board fit
            positions = [None] * len(found)
        idents = self.session.idents if self.session is not None else [None] * len(found)
        return [
            {
                "index": i,
                "diameter_mm": float(c.diameter_mm),
                "label": None if x is None else x.label,
                "type_label": None if x is None or x.label is None else idt.type_of(x.label),
                "copy": None if x is None or x.label is None else idt.copy_number(x.label),
                "name": None if x is None or x.label is None else idt.display_name(x.label),
                "score": None if x is None else _num(x.score),
                "sure": x is not None and bool(x.sure),
                "confirmed": x is not None and bool(x.confirmed),
                "wounds": self._wounds_of(None if x is None else x.label)[0],
                "max_wounds": self._wounds_of(None if x is None else x.label)[1],
                "dead": x is not None and self._is_dead(x.label),
                "position_px": p,
            }
            for i, (c, x, p) in enumerate(zip(found, idents, positions))
        ]

    def _candidates_state(self) -> list[dict[str, Any]]:
        s, i = self.session, self.selected
        if s is None or i is None:
            return []
        scores = dict(s.idents[i].candidates)
        out = []
        for label in s.menu(i):
            op = self._type_op(label)
            labels = self._copies_of(label) if op is not None and op.max > 1 else [label]
            for lab in labels:  # each copy in play of an operative that a team fields more than once
                current, maximum = self._wounds_of(lab)
                out.append({"label": lab, "name": idt.display_name(lab), "copy": idt.copy_number(lab), "score": _num(scores.get(label)), "wounds": current, "max_wounds": maximum})
        return out

    def _operatives_of(self, team: str) -> list[idt.Operative]:
        if team not in self._team_ops:
            try:
                self._team_ops[team] = idt.team_operatives(team, Path(self.gallery_root) / team)
            except (OSError, ValueError) as exc:
                self.notice.error(f"team {team}: {exc}")
                self._team_ops[team] = []
        return self._team_ops[team]

    def _team_info(self, team: str) -> idt.TeamInfo:
        if team not in self._team_infos:
            try:
                self._team_infos[team] = idt.read_team_info(Path(self.gallery_root) / team)
            except (OSError, ValueError) as exc:
                self.notice.error(f"team {team}: {exc}")
                self._team_infos[team] = idt.TeamInfo()
        return self._team_infos[team]

    def _tac_ops_of(self, slot: int) -> list[str]:
        """Tac Op choices of one side: from the archetypes of its team."""
        team = self.team_slots[slot]
        return [] if team is None else sc.tac_ops_for(self._team_info(team).archetypes)

    def _rosters_state(self) -> list[dict[str, Any] | None]:
        """The operatives of the team of each side, with the base that has
        their name in the last snapshot."""
        idents = self.session.idents if self.session is not None and self._found() else []
        out: list[dict[str, Any] | None] = []
        for slot, team in enumerate(self.team_slots):
            if team is None:
                out.append(None)
                continue
            rows = []
            for op in self._listed(slot):
                found = [(i, x) for i, x in enumerate(idents) if x.label == op.label]
                found.sort(key=lambda ix: not ix[1].confirmed)  # a confirmed name first
                i, x = found[0] if found else (None, None)
                rows.append(
                    {
                        "label": op.label,
                        "type_label": op.type_label,
                        "copy": op.copy,
                        "name": op.display,
                        "base_mm": op.base_mm,
                        "base_index": i,
                        "score": None if x is None else _num(x.score),
                        "confirmed": x is not None and bool(x.confirmed),
                        "wounds": self._wounds_of(op.label)[0],
                        "max_wounds": op.wounds,
                        "dead": self._is_dead(op.label),
                    }
                )
            in_play = self._copies_in_play()
            # Operatives that can go in the list: not in it, or with fewer copies than their `max`.
            missing = [{"label": op.label, "name": op.name, "max": op.max, "copies_in_play": in_play.get(op.label, 0)} for op in self._operatives_of(team) if in_play.get(op.label, 0) < op.max]
            out.append({"team": team, "operatives": rows, "missing": missing})
        return out

    # --- team lists -------------------------------------------------------------

    def _listed(self, slot: int) -> list[idt.Operative]:
        """Operatives in the team list of one side, in list order: one entry
        for each copy (`copy` set) of an operative with `max` > 1."""
        team = self.team_slots[slot]
        if team is None:
            return []
        ops = self._operatives_of(team)
        if self.in_play[slot] is None:
            return [replace(op, copy=1) if op.max > 1 else op for op in ops]
        by_label = {op.label: op for op in ops}
        out = []
        for label in self.in_play[slot]:
            op, k = by_label.get(idt.type_of(label)), idt.copy_number(label)
            if op is not None and (k is None) == (op.max == 1) and (k is None or k <= op.max):
                out.append(op if k is None else replace(op, copy=k))
        return out

    def _type_op(self, label: str | None) -> idt.Operative | None:
        """The team file operative of a type label or a copy label."""
        if label is None:
            return None
        t = idt.type_of(label)
        team = idt.split_label(t)[0]
        if team not in self.team_choices:
            return None
        return next((o for o in self._operatives_of(team) if o.label == t), None)

    def _copies_of(self, type_label: str) -> list[str]:
        """Labels of the copies in play of an operative type: from the team
        list of its side, else (the team is on no side) all `max` copies."""
        op = self._type_op(type_label)
        if op is None:
            return []
        if op.team in self.team_slots:
            slot = self.team_slots.index(op.team)
            return [o.label for o in self._listed(slot) if o.type_label == op.label]
        return [idt.copy_label(op.label, k) for k in range(1, op.max + 1)] if op.max > 1 else [op.label]

    def _copies_in_play(self) -> dict[str, int]:
        """Type label -> copies in the team lists (teams on a side only)."""
        out: dict[str, int] = {}
        for slot in range(TEAM_SLOTS):
            for op in self._listed(slot):
                out[op.type_label] = out.get(op.type_label, 0) + 1
        return out

    def _out_of_play(self) -> set[str]:
        """Labels of operatives that were removed from a team list (no copy in play)."""
        out: set[str] = set()
        for slot, team in enumerate(self.team_slots):
            if team is not None:
                listed = {op.type_label for op in self._listed(slot)}
                out |= {op.label for op in self._operatives_of(team) if op.label not in listed}
        return out

    def _update_roster(self) -> None:
        """Base sizes of the operatives in play (also for the b key, before
        the gallery loads), and the operatives that identification skips."""
        if any(t is not None for t in self.team_slots):
            sizes = [op.base_mm for slot in range(TEAM_SLOTS) for op in self._listed(slot)]
            self.pipe.set_roster(None if not sizes or any(s is None for s in sizes) else tuple(sorted(set(sizes))))
        else:
            self.pipe.set_roster(idt.roster_sizes(self.gallery_root, self.teams or self.team_choices))
        if self.gallery is not None:
            self.gallery.out_of_play = self._out_of_play()
            self.gallery.copies = self._copies_in_play()

    def _roster_add(self, slot: int, label: str) -> None:
        """Put an operative of the side's team at the end of its team list.
        For an operative with `max` > 1, a type label adds its lowest free
        copy; a copy label adds that copy."""
        if not self._valid_slot(slot):
            return
        team = self.team_slots[slot]
        listed = [op.label for op in self._listed(slot)]
        op = self._type_op(label) if team is not None else None
        k = idt.copy_number(label)
        if op is None or op.team != team or (op.max == 1 and k is not None) or (k is not None and k > op.max):
            self.notice.error(f"{label} cannot be added to this team list")
            return
        if op.max > 1:
            free = [idt.copy_label(op.label, n) for n in range(1, op.max + 1) if idt.copy_label(op.label, n) not in listed]
            if k is None and not free:
                self.notice.error(f"{op.name}: all {op.max} copies are in the team list")
                return
            label = free[0] if k is None else label
        if label in listed:
            self.notice.error(f"{idt.display_name(label)} cannot be added to this team list: it is in it already")
            return
        self.in_play[slot] = [*listed, label]
        self._update_roster()
        name = idt.display_name(label)
        self.notice.say(f"{name} added: push space to identify the operatives again" if self.result is not None else f"{name} added")

    def _roster_remove(self, slot: int) -> None:
        """Remove the selected operative from the team list of one side:
        identification does not give its name any more."""
        if not self._valid_slot(slot):
            return
        label = self._selected_label()
        listed = [op.label for op in self._listed(slot)]
        if label not in listed:
            self.notice.say("select an operative of this team list first")
            return
        self.in_play[slot] = [x for x in listed if x != label]
        if idt.copy_number(label) is not None:  # a copy that comes back starts again
            self.wounds.pop(label, None)
            self.positions.pop(label, None)
        self._set_selected(None)
        self._update_roster()
        self.notice.say(f"{idt.display_name(label)} removed" + (": push space to identify the operatives again" if self.result is not None else ""))

    # --- game state -------------------------------------------------------------

    def _max_wounds(self, label: str | None) -> int | None:
        """Maximum wounds of an operative label, from its team file."""
        op = self._type_op(label)
        return None if op is None else op.wounds

    def _wounds_of(self, label: str | None) -> tuple[int | None, int | None]:
        """(current, maximum) wounds of an operative label. Both None when
        the team file has no maximum. An operative starts with all its wounds."""
        maximum = self._max_wounds(label)
        if maximum is None:
            return None, None
        return min(self.wounds.get(label, maximum), maximum), maximum

    def _is_dead(self, label: str | None) -> bool:
        """An operative with 0 wounds is dead."""
        return self._wounds_of(label)[0] == 0

    def _kill_grades(self) -> list[int]:
        """Kill grade of each side: from the not expendable operatives in the
        enemy team list (starting number) and how many of them are dead."""
        grades = []
        for slot in range(TEAM_SLOTS):
            enemy = [op for op in self._listed(TEAM_SLOTS - 1 - slot) if not op.expendable]
            grades.append(sc.kill_grade(len(enemy), sum(self._is_dead(op.label) for op in enemy)))
        return grades

    def _update_kill_scores(self) -> None:
        """Kill Op of each side is automatic (`score.kill_scores`)."""
        for score, (grade, bonus) in zip(self.scores, sc.kill_scores(self._kill_grades())):
            score.kill_grade, score.kill_bonus = grade, bonus

    def _selected_label(self) -> str | None:
        """Label of the selected base, else of the operative selected in a team list."""
        s, i = self.session, self.selected
        if i is None:
            return self.selected_operative
        if s is None or i >= len(s.idents):
            return None
        return s.idents[i].label

    def _selected_wounds_state(self) -> dict[str, Any] | None:
        """Wounds of the selected base: None without a named base, `max`
        None when the team file has no maximum wounds."""
        label = self._selected_label()
        if label is None:
            return None
        current, maximum = self._wounds_of(label)
        return {"label": label, "name": idt.display_name(label), "current": current, "max": maximum}

    def _valid_slot(self, slot: int) -> bool:
        if isinstance(slot, bool) or slot not in range(TEAM_SLOTS):
            self.notice.error(f"no team slot {slot!r}")
            return False
        return True

    def _score_of(self, slot: int) -> sc.Score | None:
        return self.scores[slot] if self._valid_slot(slot) else None

    def _score_skull(self, slot: int, category: str, tp: int | None, skull: int) -> None:
        """A click on a score skull of one side (see `score.skull_value`)."""
        score = self._score_of(slot)
        if score is None:
            return
        try:
            score.click_skull(category, tp, skull)
        except ValueError as exc:
            self.notice.error(str(exc))

    def _score_tac_op(self, slot: int, name: str | None) -> None:
        """The Tac Op of one side, or None. Only a Tac Op of the team's archetypes."""
        score = self._score_of(slot)
        if score is None:
            return
        if name is not None and name not in self._tac_ops_of(slot):
            self.notice.error(f"{name!r} is not a Tac Op of the archetypes of this team")
            return
        score.tac_op = name

    def _score_primary(self, slot: int, value: str) -> None:
        score = self._score_of(slot)
        if score is None:
            return
        try:
            score.set_primary(value)
        except ValueError as exc:
            self.notice.error(str(exc))

    def _score_cp(self, slot: int, delta: int) -> None:
        """Add `delta` command points to one side (never less than 0)."""
        score = self._score_of(slot)
        if score is None:
            return
        if isinstance(delta, bool) or not isinstance(delta, int):
            self.notice.error(f"delta must be a whole number, not {delta!r}")
            return
        score.add_cp(delta)

    def _game_turn(self, delta: int) -> None:
        """Add `delta` to the turning point, kept in `score.TURNS`."""
        if isinstance(delta, bool) or not isinstance(delta, int):
            self.notice.error(f"delta must be a whole number, not {delta!r}")
            return
        self.turn = min(max(self.turn + delta, sc.TURNS[0]), sc.TURNS[-1])

    def _game_crit_op(self, name: str | None) -> None:
        """The Crit Op of the game, or None. Only a name of `score.CRIT_OPS`."""
        if name is not None and name not in sc.CRIT_OPS:
            self.notice.error(f"{name!r} is not a Crit Op")
            return
        self.crit_op = name

    def _set_wounds(self, value: int, label: str | None = None) -> None:
        """Set the current wounds of an operative (default: the selected
        base), limited to 0 .. maximum."""
        label = self._selected_label() if label is None else label
        if label is None:
            self.notice.say("select a named operative first")
            return
        name = idt.display_name(label)
        maximum = self._max_wounds(label)
        if maximum is None:
            self.notice.error(f"{name} has no maximum wounds: add wounds to team.toml (battlesight enroll --wounds)")
            return
        try:
            value = int(value)
        except (TypeError, ValueError):
            self.notice.error(f"wounds must be a whole number, not {value!r}")
            return
        self.wounds[label] = max(0, min(value, maximum))
        self.dirty = True

    def _add_wounds(self, delta: int) -> None:
        label = self._selected_label()
        current = self._wounds_of(label)[0]
        self._set_wounds(delta if current is None else current + delta)

    def _toggle_opening(self) -> None:
        """Open a closed hatchway or breach point, or close an open one."""
        i = self.selected_opening
        if i is None:
            self.notice.say("click a hatchway or a breach point first")
            return
        self.opened ^= {i}
        self.dirty = True

    # --- actions ----------------------------------------------------------------

    def dispatch(self, action_id: str, **args) -> None:
        """Do one action of `ACTIONS`. An unknown or a shell action is a KeyError.
        Each action but those of `NO_AUTOSAVE` saves the state (`autosave`)."""
        self._handlers[action_id](**args)
        if action_id not in NO_AUTOSAVE:
            self.autosave()

    def autosave(self) -> None:
        """Write the game state to the auto-save folder of this run. It never
        raises: a problem is a notice (once for the same text)."""
        if not self.autosave_enabled:
            return
        try:
            savestate.write(self, self.autosave_path, self._save_cache)
            self._save_problem = None
        except (OSError, ValueError, TypeError) as exc:
            text = f"auto-save failed: {exc}"
            if text != self._save_problem:
                self.notice.error(text)
            self._save_problem = text

    def load_state(self, path: Path | str) -> bool:
        """Restore a saved state. False (with a notice) when it cannot be read."""
        try:
            savestate.apply(self, path)
        except savestate.StateError as exc:
            self.notice.error(f"state not loaded: {exc}")
            return False
        self.notice.ok(f"state restored from {path}")
        return True

    def restore_snapshot(self, frame: np.ndarray, circles: list, idents: list | None) -> None:
        """The snapshot of a saved state: detect on `frame` again (the board
        view, the base image), then set the saved bases and names as they were,
        also the bases that the user added or removed. `idents` None: no names."""
        self._close_snapshot()
        self.win.top_down = True
        self.snap = frame.copy()
        self.frozen = True
        self.result = result = self.pipe.analyze(self.snap, source="state", until="bases")
        self.dirty = True
        img = result.scratch.get("bases_image")
        if not result.ok or img is None:
            self.notice.error(f"saved snapshot: {failed_stage(result)}")
            return
        ref = self.pipe.cache.get("bases_reference")
        crops = base_crops(img, circles, self.cfg)
        backgrounds = [] if ref is None else base_crops(ref, circles, self.cfg)
        result.scratch.update(bases=list(circles), base_crops=crops, base_backgrounds=backgrounds, idents=idents if idents is not None else [])
        if idents is None:
            return
        if not self.gallery_tried:
            self._load_gallery()
        if self.gallery is None:
            self.notice.error("saved names not shown: no gallery")
            return
        self.session = self._label_session(result, self.gallery)
        if self.session is not None and len(self.session.idents) != len(circles):
            raise ValueError("saved bases and names differ in count")

    def _need_frame(self) -> bool:
        if self.frame is None:
            self.notice.error("no camera frame yet")
            return False
        return True

    def _close_snapshot(self) -> None:
        self.result = self.snap = self.session = self.shown = None
        self.selected = self.target = self.ring_in = self.selected_opening = self.new_base = None
        self.adding = self.shooting = self.locating_light = False
        self.frozen = False

    def _load_gallery(self) -> None:
        self.gallery_tried = True
        teams = self.teams or idt.list_teams(self.gallery_root)
        if not teams:
            self.notice.error(f"no gallery teams in {self.gallery_root}: battlesight enroll")
            return
        self.gallery = load_gallery(teams, self.gallery_root, self.cfg)
        if self.gallery is None:
            self.notice.error("gallery: cannot load, see the Terminal")
        else:
            self.gallery.out_of_play = self._out_of_play()
            self.gallery.copies = self._copies_in_play()
        self.pipe.set_gallery(self.gallery)

    def _snapshot(self, name: bool) -> None:
        if name and not self._need_model():
            return
        if not self._need_frame():
            return
        with self._task(TASK_TEXT["snapshot.name" if name else "snapshot.bases"]) as step:
            self._run_snapshot(self.frame, name, step)

    def _run_snapshot(self, frame: np.ndarray, name: bool, step: Callable[[int, int], None] = lambda _d, _t: None) -> None:
        """Steps: the gallery load (the first time), each stage, the name session."""
        load = name and not self.gallery_tried
        before, after = int(load), int(name)
        if load:
            step(0, before + len(self.pipe.names) + after)
            self._load_gallery()
        self._close_snapshot()
        self.win.top_down = True  # a snapshot opens top-down, the v key changes it
        self.snap = frame.copy()
        self.frozen = True
        self.result = result = self.pipe.analyze(self.snap, source="camera", until=None if name else "bases", progress=lambda done, total: step(before + done, before + total + after))
        print(result.summary())
        gallery = self.gallery
        if name and gallery is not None and result.ok:
            self.session = self._label_session(result, gallery)
            if self.session is not None:
                self._pair_copies()
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
        total = before + len(result.stages) + after
        step(total, total)
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
        """A new selected base starts with no ring and no second base. Only
        one thing is selected: a base or an opening."""
        self.selected = index
        self.target = self.ring_in = self.selected_opening = self.selected_operative = self.selected_light = None
        self.shooting = False
        if index != self.new_base:
            self.new_base = None
        self.dirty = True

    def _select_opening(self, index: int | None) -> None:
        self._set_selected(None)
        self.selected_opening = index

    def _deselect(self) -> None:
        self._set_selected(None)

    def _select_index(self, index: int) -> None:
        """Select base `index` of the last snapshot (for example from a team list)."""
        if not 0 <= index < len(self._found()):
            self.notice.error(f"no base {index}")
            return
        self._set_selected(index)

    def _select_operative(self, label: str) -> None:
        """Select an operative of a team list: its base when the snapshot
        has its name (a confirmed name first), else the operative alone
        (name and wounds, no base)."""
        if not any(op.label == label for slot in range(TEAM_SLOTS) for op in self._listed(slot)):
            self.notice.error(f"{label} is not in a team list")
            return
        idents = self.session.idents if self.session is not None and self._found() else []
        found = sorted((i for i, x in enumerate(idents) if x.label == label), key=lambda i: not idents[i].confirmed)
        self._set_selected(found[0] if found else None)
        if not found:
            self.selected_operative = label

    def _back(self) -> None:
        """Go back to the live video, from the snapshot or the minimap. The
        bases, names and selection of the snapshot stay, in board mm, until
        the next snapshot. In the live
        video the key clears the selection. While a base is added, it only
        stops that."""
        if self.adding or self.locating_light:
            self.adding = self.locating_light = False
        elif self.shooting:
            self._toggle_shooting()
        elif self.frozen or self._minimap_on():
            self._show_live()
        else:
            self._deselect()

    def _show_live(self) -> None:
        """The live video. The bases, names and selection of the snapshot stay."""
        self.frozen = self.minimap = False
        self.dirty = True

    def _show_snapshot(self) -> None:
        """The last snapshot again. With no snapshot yet, the video stops on
        the current frame: a snapshot with no bases."""
        if self.result is None:
            if not self._need_frame():
                return
            self._still_snapshot()
        self.minimap = False
        self.frozen = True
        self.dirty = True

    def _still_snapshot(self) -> bool:
        """A snapshot of the current frame with no bases and no names: only
        the board fit (corners, rectify). False when the board fit failed;
        the frame stays on the screen."""
        self._close_snapshot()
        self.win.top_down = True
        self.snap = self.frame.copy()
        self.frozen = True
        self.result = result = self.pipe.analyze(self.snap, source="camera", until="rectify")
        self.dirty = True
        if not result.ok:
            self.notice.error(failed_stage(result))
            return False
        img = bases_image(self.snap, result.scratch["view"], self.cfg)
        result.scratch.update(bases=[], bases_image=img, idents=[], base_crops=[], base_backgrounds=[])
        return True

    def _toggle_view(self) -> None:
        """The live view and the snapshot each have their own view."""
        if self.frozen:
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
        self._read_pieces()

    def _read_pieces(self) -> None:
        """The piece files of the terrain type: the folder under
        `pieces_root` with the name of the folder of its map files."""
        entry = next((e for e in self.maps if e.terrain == self.terrain_type), None)
        self.pieces, problems = ({}, []) if entry is None else ter.load_pieces(self.pieces_root / entry.path.parent.name)
        for text in problems:
            self.notice.error(f"piece file not used: {text}")

    def _build_shapes(self) -> None:
        """The shapes of the layout, the light terrain at its poses. The
        openings keep their order, so `opened` stays valid."""
        self.terrain_shapes = [] if self.terrain_map is None else ter.shapes(self.terrain_map, self.cfg, self.pieces, self.light_poses)
        for key, center, angle in self.loose_light:  # after the light items of the layout: `selected_light`
            self.terrain_shapes.append(ter.Shape("light", self.pieces[key].at(center, angle), center, name=key, located=True))
        self.terrain_shapes += [ter.Shape("light_key", np.array([center]), center, name=key) for key, center, _ in self.loose_light]
        self._cover_card_symbols()
        self.openings = [s for s in self.terrain_shapes if s.kind in OPENING_KINDS]
        self._light_version += 1
        self.dirty = True

    def _cover_card_symbols(self) -> None:
        """A card symbol (and its key) with an identified piece on it, for
        example a piece of another key or of "Locate terrain": `covered`.
        The light shapes and the light keys are in the same order."""
        light = [i for i, s in enumerate(self.terrain_shapes) if s.kind == "light"]
        keys = [i for i, s in enumerate(self.terrain_shapes) if s.kind == "light_key"]
        found = [self.terrain_shapes[i] for i in light if self.terrain_shapes[i].located]

        def inside(point, poly) -> bool:
            return cv2.pointPolygonTest(poly.points.astype(np.float32).reshape(-1, 1, 2), (float(point[0]), float(point[1])), False) >= 0

        for n, i in enumerate(light):
            card = self.terrain_shapes[i]
            if card.located or not any(inside(f.center, card) or inside(card.center, f) for f in found):
                continue
            self.terrain_shapes[i] = replace(card, covered=True)
            if n < len(keys):
                self.terrain_shapes[keys[n]] = replace(self.terrain_shapes[keys[n]], covered=True)
            if self.selected_light == n:
                self.selected_light = None

    # --- light terrain on the board -------------------------------------------------

    def _light_image(self) -> np.ndarray | None:
        """The top-down image for the search: of the snapshot while it is on
        the screen, else of the newest frame."""
        if self.frozen and self.result is not None and self.result.scratch.get("bases_image") is not None:
            return self.result.scratch["bases_image"]
        if self.frame is None or self.view is None:
            return None
        return bases_image(self.frame, self.view, self.cfg)

    def _locate_light(self) -> None:
        """"Identify all terrain": search each light terrain piece of the
        layout that has a piece file and no pose yet. A piece that was
        found keeps its pose (later a model can hide it) until "Remove ID"."""
        if self.terrain_map is None:
            self.notice.say("choose a layout first")
            return
        todo = [k for k, item in enumerate(self.terrain_map.light) if item.key in self.pieces and k not in self.light_poses]
        no_file = sorted({item.key for item in self.terrain_map.light if item.key not in self.pieces})
        image = self._light_image() if todo else None
        if todo and image is None:
            self.notice.error("no board in the image: the terrain cannot be identified")
            return
        with self._task(TASK_TEXT["light.locate"]) as step:
            missed = self._search_layout_light(todo, image, step)
        if todo:
            self._build_shapes()
        located = sum(1 for k, item in enumerate(self.terrain_map.light) if k in self.light_poses)
        text = f"light terrain: {located} of {len(self.terrain_map.light)} identified"
        if missed:
            text += f"; not found: {', '.join(missed)}"
        if no_file:
            text += f"; no outline file: {', '.join(no_file)}"
        (self.notice.say if missed or no_file else self.notice.ok)(text)

    def _search_layout_light(self, todo: list[int], image: np.ndarray | None, step: Callable[[int, int], None]) -> list[str]:
        """Search the light items `todo` of the layout (one step each). Returns the keys that were not found."""
        scale = self.cfg.bases_px_per_mm
        missed = []
        for n, k in enumerate(todo):
            step(n, len(todo))
            item = self.terrain_map.light[k]
            pose = lightfit.locate(
                image,
                scale,
                (-BASES_MARGIN_MM, -BASES_MARGIN_MM),
                self.pieces[item.key].outline,
                ter.light_rect(item, self.cfg, self.cfg.light_margin_mm),  # the card footprint, with a margin
                item.at,
                item.angle_deg,
                min_score=self.cfg.light_min_score,
            )
            if pose.confirmed:
                self.light_poses[k] = (pose.center_mm, pose.angle_deg)
            else:
                missed.append(item.key)
            print(f"  light terrain {item.key}: {'found' if pose.confirmed else 'not found'}, score {pose.score:.2f}, at ({pose.center_mm[0]:.0f}, {pose.center_mm[1]:.0f}) mm, {pose.angle_deg:.0f} deg")
        step(len(todo), len(todo))
        return missed

    def _remove_light(self) -> None:
        """"Remove ID": the selected piece goes back to the card symbol."""
        k, items = self.selected_light, self._light_items()
        if k is not None and k >= len(items) and k - len(items) < len(self.loose_light):
            del self.loose_light[k - len(items)]  # no item of the layout: the piece goes away
            self.selected_light = None
        elif k is not None and k in self.light_poses:
            del self.light_poses[k]
        else:
            self.notice.say("click an identified light terrain piece first")
            return
        self._build_shapes()

    def _light_items(self) -> tuple[ter.Light, ...]:
        return () if self.terrain_map is None else tuple(self.terrain_map.light)

    def _start_locating_light(self) -> None:
        """"Locate terrain": the next click is on a light terrain piece on
        the board. `Esc` stops it. It needs no layout."""
        if not self.pieces:
            self.notice.error("no outline file of a light terrain piece")
            return
        self._locate_image = self._light_image()  # also the pointer crop: one board warp
        if self._locate_image is None:
            self.notice.error("no board in the image: the terrain cannot be located")
            return
        self.adding = False
        self.locating_light = True
        self._set_selected(None)
        self.notice.say("click the terrain piece, Esc to cancel")

    def _locate_light_at(self, x: float, y: float) -> None:
        """The click of "Locate terrain" at pixel (x, y): `_locate_light_mm`."""
        self.locating_light = False
        at = self._to_mm(x, y)
        if at is None:
            print("locate terrain: no board position under the click")
            return
        image, self._locate_image = self._locate_image, None
        self._locate_light_mm((float(at[0]), float(at[1])), image)

    def _locate_selected_light(self) -> None:
        """"Identify terrain here": "Locate terrain" at the centre of the
        selected card symbol. The found piece becomes the selection."""
        k = self.selected_light
        items = self._light_state()
        if k is None or k >= len(items) or items[k]["located"]:
            self.notice.say("click a light terrain piece that is not identified first")
            return
        found = self._locate_light_mm(tuple(items[k]["center_mm"]))
        if found is not None:
            self._select_light(found)

    def _locate_light_mm(self, at: tuple[float, float], image: np.ndarray | None = None) -> int | None:
        """Search each piece file around the board point `at`
        (`lightfit.locate_at`: only poses with the point in the outline).
        The best piece that is found must have a score `[light] min_margin`
        above the next piece. Else nothing changes, a notice tells why and
        the scores go to standard output. With a layout, the piece becomes the
        nearest item of its key that has no pose; else a piece of its own
        (`loose_light`). Returns the index of the found piece in the light
        shapes (as `selected_light`), else None. `image`: the top-down image
        to search, else `_light_image`."""
        image = self._light_image() if image is None else image
        if image is None:
            self.notice.error("no board in the image: the terrain cannot be located")
            return None
        if any(cv2.pointPolygonTest(s.points.astype(np.float32).reshape(-1, 1, 2), at, False) >= 0 for s in self.terrain_shapes if s.kind == "light" and s.located):
            print("locate terrain: the piece there is identified already")
            return None
        with self._task(TASK_TEXT["light.locate_at"]) as step:
            res = lightfit.locate_at(image, self.cfg.bases_px_per_mm, (-BASES_MARGIN_MM, -BASES_MARGIN_MM), {k: p.outline for k, p in self.pieces.items()}, at, self.cfg.light_min_score, self.cfg.light_min_margin, progress=step)
        print("  locate terrain: " + ", ".join(f"{k} {v:.2f}{' found' if k in res.found else ''}" for k, v in res.scores.items()))
        if not res.found:
            print(f"locate terrain: no piece found at ({at[0]:.0f}, {at[1]:.0f}) mm")
            self.notice.say("no light terrain piece found there: click the middle of the piece")
            return None
        if res.key is None:
            first, second = res.found[:2]
            print(f"locate terrain: no clear result at ({at[0]:.0f}, {at[1]:.0f}) mm: {first} {res.scores[first]:.2f}, {second} {res.scores[second]:.2f}")
            self.notice.say(f"light terrain not sure: {first} or {second}")
            return None
        key, pose = res.key, res.pose
        free = [k for k, item in enumerate(self._light_items()) if item.key == key and k not in self.light_poses]
        if free:
            k = min(free, key=lambda k: float(np.hypot(*np.subtract(self.terrain_map.light[k].at, pose.center_mm))))
            self.light_poses[k] = (pose.center_mm, pose.angle_deg)
        else:
            self.loose_light.append((key, pose.center_mm, pose.angle_deg))
            k = len(self._light_items()) + len(self.loose_light) - 1
        self._build_shapes()
        self.notice.ok(f"light terrain {key} located")
        return k

    def _light_state(self) -> list[dict[str, Any]]:
        """Each light terrain piece: `located`: found on the board (real
        outline at `center_mm`, `angle_deg`; else the card symbol at the
        place of the layout card); `outline`: it has a piece file."""
        out = []
        for k, item in enumerate(self._light_items()):
            center, angle = self.light_poses.get(k, (item.at, item.angle_deg))
            out.append({"index": k, "key": item.key, "center_mm": [round(float(center[0]), 1), round(float(center[1]), 1)], "angle_deg": round(float(angle), 1), "located": k in self.light_poses, "outline": item.key in self.pieces})
        for key, center, angle in self.loose_light:
            out.append({"index": len(out), "key": key, "center_mm": [round(float(center[0]), 1), round(float(center[1]), 1)], "angle_deg": round(float(angle), 1), "located": True, "outline": True})
        return out

    def _light_under(self, x: float, y: float) -> int | None:
        """The light terrain piece at pixel (x, y) of the returned image. A
        covered card symbol is not clicked."""
        shapes = [s for s in self.terrain_shapes if s.kind == "light"]
        p_mm = self._to_mm(x, y) if shapes else None
        if p_mm is None:
            return None
        for k, shape in enumerate(shapes):
            if not shape.covered and cv2.pointPolygonTest(shape.points.astype(np.float32).reshape(-1, 1, 2), (float(p_mm[0]), float(p_mm[1])), False) >= 0:
                return k
        return None

    def _select_light(self, index: int | None) -> None:
        self._set_selected(None)
        self.selected_light = index

    def _shapes_now(self) -> list[ter.Shape]:
        """The shapes of the layout, the open openings with `open` set."""
        opened = {id(self.openings[i]) for i in self.opened}
        return [replace(s, open=True) if id(s) in opened else s for s in self.terrain_shapes]

    def _draw_terrain(self, img: np.ndarray, to_px, always: bool = False) -> None:
        """`always`: the layout shows also with "Layout" off (the minimap)."""
        on = self.terrain or always
        switch = {"zone": self.zones, "wall_key": self.keys, "teleporter_key": self.keys, "light_key": self.keys and self.light_shown, "light": self.light_shown and (on or self.terrain_map is None)}  # no layout: only pieces of "Locate terrain"
        shown = [s for s in self._shapes_now() if switch.get(s.kind, on) and not s.covered]
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

    def _toggle_light(self) -> None:
        self.light_shown = not self.light_shown
        self.dirty = True

    def _show_minimap(self) -> None:
        """The minimap over the live video or the snapshot. It needs a layout."""
        if self.map_layout is None:
            self.minimap = False
            self.notice.say("choose a layout first")
            return
        self.minimap = True
        self.dirty = True

    def _set_terrain_type(self, name: str) -> None:
        """A new terrain type starts with no layout."""
        if name not in {e.terrain for e in self.maps}:
            self.notice.error(f"no maps of terrain type {name!r} in {self.maps_root}")
            return
        if name != self.terrain_type:
            self.terrain_type = name
            self._read_pieces()
            self._set_map_layout(None)

    def _set_map_layout(self, name: str | None) -> None:
        """Load the layout `name` of the terrain type, or no layout for None.
        A bad file gives a notice and leaves no layout. Without a layout the
        minimap goes off."""
        self._load_map_layout(name)
        if self.map_layout is None:
            self.minimap = False

    def _load_map_layout(self, name: str | None) -> None:
        self.dirty = True
        self.map_layout, self.terrain_shapes, self.openings, self.selected_opening = None, [], [], None
        self.terrain_map, self.light_poses, self.loose_light = None, {}, []  # the pieces found on the board belong to the layout that was on
        self.selected_light = None
        self._light_version += 1
        self.opened = set()  # a new layout starts with all hatchways and breach points closed
        self._navigator = self._wall_path = self._wall_range = self._move_field = self._move_route = None
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
        self.terrain_map = tmap
        self.map_layout, self.terrain = name, True
        self._build_shapes()
        self.around_walls = True  # the rule of the terrain: no measurement through walls
        self.notice.ok(f"map {tmap.terrain} / {tmap.name}")

    # --- teams ------------------------------------------------------------------

    def _set_team(self, slot: int, name: str | None) -> None:
        """The team of one side, or None. The gallery loads again at the next
        identification; the names of the last snapshot stay until then."""
        if slot not in range(TEAM_SLOTS):
            self.notice.error(f"no team slot {slot}")
            return
        if name is not None and name not in self.team_choices:
            self.notice.error(f"no team {name!r} in {self.gallery_root}")
            return
        if name is not None and name in self.team_slots and self.team_slots[slot] != name:
            self.notice.error(f"{name} is on the other side already")
            return
        if self.team_slots[slot] == name:
            return
        old = self.team_slots[slot]
        if old is not None:  # the copies of the old team are gone
            self.positions = {k: v for k, v in self.positions.items() if idt.split_label(k)[0] != old}
        self.team_slots[slot] = name
        self.teams = [t for t in self.team_slots if t] or None
        if self.scores[slot].tac_op not in self._tac_ops_of(slot):  # not a Tac Op of the new team
            self.scores[slot].tac_op = None
        self.in_play[slot] = None  # a new team starts with all its operatives
        self.gallery, self.gallery_tried = None, False
        self.pipe.set_gallery(None)
        self._update_roster()
        self.notice.say("teams changed: push space to identify the operatives again" if self.result is not None else f"teams: {', '.join(self.teams or ['all'])}")

    def _take_reference(self) -> None:
        if not self._need_frame():
            return
        try:
            self.pipe.set_reference(self.frame)
            self.reference_frame = self.frame.copy()
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
        self.reference_frame = None
        self.notice.say("empty-board reference cleared: single-frame detection")

    def _base_under(self, x: float, y: float) -> int | None:
        """The base at pixel (x, y) of the returned image."""
        found = self._found()
        p_mm = self._to_mm(x, y) if found else None
        if p_mm is None:
            return None
        return measure.base_at([c.center_mm for c in found], [c.diameter_mm for c in found], p_mm)

    def _opening_under(self, x: float, y: float) -> int | None:
        """The hatchway or breach point at pixel (x, y) of the returned image."""
        p_mm = self._to_mm(x, y) if self.openings else None
        if p_mm is None:
            return None
        best, index = -OPENING_HIT_MM, None
        for i, sh in enumerate(self.openings):
            d = cv2.pointPolygonTest(sh.points.astype(np.float32).reshape(-1, 1, 2), (float(p_mm[0]), float(p_mm[1])), True)
            if d >= best:
                best, index = d, i
        return index

    def _click(self, x: float, y: float) -> None:
        """A left click selects the base there (a click on the selected base
        takes it away). Without a base there, a click on a hatchway or a
        breach point selects it. A right click measures (`_select_for_names`)."""
        if not self._mapped():
            return
        if self.locating_light:
            self._locate_light_at(x, y)
            return
        if self.adding:
            self._add_at(x, y)
            return
        index = self._base_under(x, y)
        if index is None:
            opening = self._opening_under(x, y)
            light = self._light_under(x, y) if opening is None else None
            if light is not None:
                self._select_light(None if light == self.selected_light else light)
            else:
                self._select_opening(None if opening == self.selected_opening else opening)
        else:
            self._set_selected(None if index == self.selected else index)

    def _select_for_names(self, x: float, y: float) -> None:
        """A right click. With a selected base, a right click on another base
        makes it the target of the distance (again: takes it away). Else the
        base there becomes the selected base, to fix its name (a front end
        can open its name popup: `right_click` is "name")."""
        self.right_click = None
        if not self._mapped():
            return
        index = self._base_under(x, y)
        if index is None:
            self._select_opening(self._opening_under(x, y))
            return
        if self.selected is not None and index != self.selected:
            self.target = None if index == self.target else index
            self.right_click = "measure"
            self.dirty = True
            return
        self.right_click = "name"
        self._set_selected(index)
        s = self.session
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
        popup does not matter. Base `index` is selected after it."""
        s = self.session
        index = self.selected if index is None else index
        if s is None or index is None:
            return
        label = self._copy_for(label, index)
        if label is None:
            return
        old = s.idents[index].label
        path = s.choose(label, index)
        found = self._found()
        for k, x in enumerate(s.idents):  # the chosen copy and the copy of a swap are where the user says
            if x.label is not None and idt.copy_number(x.label) is not None and x.label in (label, old) and k < len(found):
                self.positions[x.label] = np.asarray(found[k].center_mm, float)
        self.new_base = None
        self._set_selected(index)  # the named base stays selected
        name = idt.display_name(label)
        if path is None:
            print(f"named {label}: the gallery has this photo already, no photo added")
            self.notice.ok(f"named {name}, the gallery has this photo already")
        else:
            print(f"named {label}: added {path}")
            self.notice.ok(f"named {name}, photo added to the gallery")
        self.dirty = True

    def _copy_for(self, label: str, index: int) -> str | None:
        """The label that a name choice gives base `index`. For a type label of
        an operative with `max` > 1: the copy that the base has, else a copy
        in play that has no base, else the first copy (the bases swap). None
        (with a notice) when no copy is in play."""
        op = self._type_op(label)
        if op is None or op.max == 1:
            return label
        copies = self._copies_of(op.label)
        if idt.copy_number(label) is not None:
            if label not in copies:
                self.notice.error(f"{idt.display_name(label)} is not in a team list")
                return None
            return label
        if not copies:
            self.notice.error(f"no {op.name} in the team list: add one first")
            return None
        mine = self.session.idents[index].label
        if mine in copies:
            return mine
        held = {x.label for x in self.session.idents}
        free = [c for c in copies if c not in held]
        return (free or copies)[0]

    # --- copies ---------------------------------------------------------------------

    def _pair_copies(self) -> None:
        """Give each base with the type label of an operative with `max` > 1
        one copy of it: the pairing with the shortest total distance to the
        last known positions of the copies (`positions`). A swap of two
        copies that costs less than `copies_swap_margin_mm` makes both names
        unsure when the copies have a different state. The positions of the
        sure bases change."""
        from scipy.optimize import linear_sum_assignment

        s, found = self.session, self._found()
        by_type: dict[str, list[int]] = {}
        for i, x in enumerate(s.idents):
            op = self._type_op(x.label)
            if op is not None and op.max > 1 and idt.copy_number(x.label) is None:
                by_type.setdefault(op.label, []).append(i)
        new_cost, margin = self.cfg.copies_new_cost_mm, self.cfg.copies_swap_margin_mm
        for t, bases in by_type.items():
            copies = self._copies_of(t)
            if not copies:
                continue
            centers = np.array([found[i].center_mm for i in bases], float).reshape(-1, 2)

            def dist(b: int, c: str) -> float:
                p = self.positions.get(c)
                return new_cost if p is None else float(np.linalg.norm(centers[b] - p))

            cost = np.array([[dist(b, c) for c in copies] for b in range(len(bases))])
            rows, cols = linear_sum_assignment(cost)
            pick = dict(zip(rows.tolist(), cols.tolist()))
            # Copies with no position (first snapshot): lowest number to the base nearest the top-left corner.
            new = sorted((b for b, c in pick.items() if copies[c] not in self.positions), key=lambda b: float(np.hypot(*centers[b])))
            for b, c in zip(new, sorted(pick[b] for b in new)):
                pick[b] = c
            on_base = {c: b for b, c in pick.items()}
            state = [self._wounds_of(c)[0] for c in copies]
            unsure: set[int] = set()
            for b, c in pick.items():
                for c2 in range(len(copies)):
                    if c2 == c or state[c2] == state[c]:
                        continue
                    b2 = on_base.get(c2)
                    extra = dist(b, copies[c2]) - dist(b, copies[c])
                    if b2 is not None:
                        extra += dist(b2, copies[c]) - dist(b2, copies[c2])
                    if extra < margin:
                        unsure.add(b)
                        if b2 is not None:
                            unsure.add(b2)
            for b, i in enumerate(bases):
                x = s.idents[i]
                if b not in pick:  # more bases than copies in play: the name stays the type
                    continue
                label = copies[pick[b]]
                s.idents[i] = idt.Ident(label, x.score, x.margin, x.candidates, sure=x.sure and b not in unsure, confirmed=False, masked=x.masked)
                if b not in unsure:
                    self.positions[label] = centers[b].copy()

    def _remove_base(self, index: int | None = None) -> None:
        """Remove base `index` (default: the selected base) from the snapshot:
        a false base, for example terrain or board print. The later bases move
        one index down. Gallery photos that a name fix added for it go too."""
        found = self._found()
        index = self.selected if index is None else index
        if index is None or not 0 <= index < len(found):
            self.notice.say("click a base first")
            return
        scratch = self.result.scratch
        scratch["bases"] = [c for k, c in enumerate(found) if k != index]
        for key in ("idents", "base_crops", "base_backgrounds"):
            items = scratch.get(key)
            if items is not None and index < len(items):
                scratch[key] = [x for k, x in enumerate(items) if k != index]
        removed = self.session.remove(index) if self.session is not None else []
        self._set_selected(None)
        for path in removed:
            print(f"removed {path}")
        self.notice.say("base removed" + (f", {len(removed)} gallery photo removed" if len(removed) == 1 else f", {len(removed)} gallery photos removed" if removed else ""))
        self.dirty = True

    def _start_adding(self) -> None:
        """"Locate operative": the next click is the centre of a base that the
        detector missed. `Esc` stops it."""
        if self.adding or not self._need_model():
            return
        self.locating_light = False
        if not self.frozen and self.session is None and not self._need_frame():
            return
        self.busy = True
        try:
            if self.on_busy is not None:
                self.on_busy(self.current())
            ready = self._prepare_adding()
        finally:
            self.busy = False
        if ready:
            self.adding = True
            self._set_selected(None)
            self.notice.say("click the centre of the base, Esc to cancel")

    def _prepare_adding(self) -> bool:
        """A snapshot with a name session for "Locate operative". In the live
        video, the names of the last identification come back with their
        snapshot; with no names yet, a new snapshot of the frame with no bases
        (the user adds each one). A snapshot of bases only (`b`) gets a name
        session with its bases unknown."""
        if not self.gallery_tried:
            self._load_gallery()
        gallery = self.gallery
        if gallery is None:
            self.notice.error("no gallery: cannot name an operative")
            return False
        if not self.frozen and self.session is not None:
            self.frozen = True
            self.win.top_down = True
            self.dirty = True
            return True
        if not self.frozen and not self._still_snapshot():
            return False
        result = self.result
        if result is None or not result.ok or result.scratch.get("bases_image") is None:
            self.notice.error("no board in the snapshot: push space")
            return False
        if self.session is None:
            found = self._found()
            img, ref = result.scratch["bases_image"], self.pipe.cache.get("bases_reference")
            result.scratch["base_crops"] = base_crops(img, found, self.cfg)
            result.scratch["base_backgrounds"] = [] if ref is None else base_crops(ref, found, self.cfg)
            self.session = self._label_session(result, gallery)
            self.dirty = True
        return True

    def _add_at(self, x: float, y: float) -> None:
        """Search a base (also a part of its rim) around the clicked point of
        the snapshot, add it and name it as the gallery says. The new base is
        selected, and the front end shows its name list."""
        s, gallery = self.session, self.gallery
        at = self._to_mm(x, y)
        img = self.result.scratch.get("bases_image") if s is not None and gallery is not None else None
        if at is None or img is None:
            self.adding = False
            return
        found = self._found()
        with self._task(TASK_TEXT["base.add"]) as step:
            circle = find_base_near(img, self.cfg, at, gallery.base_sizes(), [(c.center_mm, c.diameter_mm) for c in found])
            step(1, 2)
            if circle is not None:
                crop = base_crops(img, [circle], self.cfg)[0]
                ref = self.pipe.cache.get("bases_reference")
                background = None if ref is None else base_crops(ref, [circle], self.cfg)[0]
                ident = idt.identify([crop], [circle.diameter_mm], gallery, [] if background is None else [background], [circle.center_mm])[0] if len(gallery) else None
                step(2, 2)
        if circle is None:
            self.notice.error("no base found there: click the centre of the base")  # self.adding stays on, the user clicks again
            return
        self.adding = False
        if ident is not None:
            op = self._type_op(ident.label)
            if op is not None and op.max > 1:  # a copy that has no base, else the user decides
                held = {x.label for x in s.idents}
                free = [c for c in self._copies_of(op.label) if c not in held]
                ident = replace(ident, label=free[0]) if free else idt.Ident(None, ident.score, -np.inf, ident.candidates)
            elif ident.label is not None and any(x.label == ident.label for x in s.idents):
                ident = idt.Ident(None, ident.score, -np.inf, ident.candidates)  # the name is on another base: the user decides
        else:
            ident = idt.Ident(None, float("-inf"), float("-inf"))
        index = s.add(circle.center_mm, circle.diameter_mm, crop, ident, background)
        if ident.label is not None and idt.copy_number(ident.label) is not None:
            self.positions[ident.label] = np.asarray(circle.center_mm, float)
        scratch = self.result.scratch
        scratch["bases"] = [*found, circle]
        for key, item in (("idents", ident), ("base_crops", crop), ("base_backgrounds", background)):
            if scratch.get(key) is not None and item is not None:
                scratch[key] = [*scratch[key], item]
        self._set_selected(index)
        self.new_base = index
        if not s.menu(index):
            self.notice.error(f"no operative with a {circle.diameter_mm:g} mm base in the gallery: battlesight enroll")
        else:
            self.notice.say(f"base found ({circle.diameter_mm:g} mm): choose the operative")

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

    def _save_state(self) -> None:
        """"Save state": the game state in its own folder `saves_dir/<time>_turn<n>`."""
        base = self.saves_dir / f"{time.strftime('%Y-%m-%d_%H%M')}_turn{self.turn}"
        folder, n = base, 1
        while folder.exists():
            n += 1
            folder = base.with_name(f"{base.name}-{n}")
        try:
            savestate.write(self, folder)
        except (OSError, ValueError, TypeError) as exc:
            self.notice.error(f"state not saved: {exc}")
            return
        print(f"saved state {folder}")
        self.notice.ok(f"state saved: {folder.name}")

    def _set_ui_layout(self, name: str) -> None:
        """The front end tells its layout, so a saved state keeps it."""
        self.ui_layout = name
