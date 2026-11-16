"""Terrain map files: walls, pillars, openings, objectives, teleporters, zones, light terrain.

A map file (`data/maps/<terrain>/<layout>.toml`) holds one layout of a
terrain type. An agent writes it from a photo of the layout card (skill
`record-map`). Points are grid intersections `at = [col, row]` (steps of 0.5)
or board mm `at_mm = [x, y]`. `load_map` checks the file and turns all points
into board mm. `shapes` gives the footprints on the board for the overlay,
with the sizes of `[terrain]` in `data/board.toml`. No window, no camera.

A piece file (`data/terrain/<folder>/<key>.toml`, skill `record-terrain`)
holds the real outline of one light terrain piece: a polygon in the piece
frame (origin = centre of its bounding box, x = long side).
With a piece file, `shapes` gives that polygon in place of the card symbol,
at the pose found on the board (`poses`, `lightfit`).
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import BoardConfig

DEFAULT_MAPS_ROOT = Path(__file__).resolve().parents[2] / "data" / "maps"
DEFAULT_PIECES_ROOT = Path(__file__).resolve().parents[2] / "data" / "terrain"
SCHEMA = 1
PIECE_SCHEMA = 1
OPENINGS = ("none", "hatchway", "breach")
WALL_KEY = re.compile(r"[AB][1-9][0-9]?$")
LIGHT_EDGE_TOL_MM = 1.0  # light terrain may stick out of the board this much (it touches the edge)


class MapError(ValueError):
    """A map file that cannot be used. The message names the file and the problem."""


@dataclass(frozen=True)
class Wall:
    start: tuple[float, float]  # board mm, pillar centre
    end: tuple[float, float]
    grids: int  # length in grid sections: 2 = A wall, 1 = B wall
    opening: str = "none"
    opening_at: tuple[float, float] | None = None  # board mm, centre of the opening
    key: str = ""  # wall element on the card: A1..A4 (2 grid), B1..B4 (1 grid)


@dataclass(frozen=True)
class WallEnd:
    at: tuple[float, float]  # board mm, the grid point at the end of the wall
    toward: tuple[float, float]  # unit vector from the point along the wall


@dataclass(frozen=True)
class Light:
    key: str  # light terrain on the card: C1..C5
    at: tuple[float, float]  # board mm, centre of the footprint
    long_axis: str = "x"  # "x" or "y": the long side of the footprint is along this board axis

    @property
    def angle_deg(self) -> float:
        """Angle of the long side from the board x axis, as the card shows it."""
        return 0.0 if self.long_axis == "x" else 90.0


@dataclass(frozen=True)
class Piece:
    """The real outline of one light terrain piece (piece file)."""

    key: str
    outline: np.ndarray  # (N, 2) mm, piece frame: origin = centre of the bounding box, x = long side

    def at(self, center, angle_deg: float) -> np.ndarray:
        """The outline (N, 2) in board mm at a pose."""
        a = np.radians(angle_deg)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        return self.outline @ rot.T + np.asarray(center, float)


@dataclass(frozen=True)
class Zone:
    name: str
    rect: tuple[float, float, float, float]  # x0, y0, x1, y1 in board mm


@dataclass(frozen=True)
class TerrainMap:
    name: str
    terrain: str
    source: str
    path: Path | None
    walls: tuple[Wall, ...] = ()
    pillars: tuple[tuple[float, float], ...] = ()
    wall_ends: tuple[WallEnd, ...] = ()
    objectives: tuple[tuple[float, float], ...] = ()
    teleporters: tuple[tuple[float, float], ...] = ()
    zones: tuple[Zone, ...] = ()
    light: tuple[Light, ...] = ()


@dataclass(frozen=True)
class Shape:
    """One footprint in board mm. `kind`: zone, wall, hatchway, breach,
    pillar, wall_end, objective, objective_range (the circle 1 in around
    an objective marker), teleporter, light (light terrain C1..C5, `name` = key),
    wall_key (label of a wall at its middle), teleporter_key (`T` at the pad
    centre), light_key (`C1`.. at the centre of the light terrain). Keys come last. A polygon has `points` (N, 2); a
    disc has `center` and `radius` (and `points` sampled on its rim)."""

    kind: str
    points: np.ndarray
    center: tuple[float, float] | None = None
    radius: float | None = None
    name: str = field(default="")
    vertical: bool = False  # wall_key: the label of a vertical wall goes to its right
    located: bool = False  # light: the real outline at the pose found on the board (else the card symbol)
    open: bool = False  # hatchway or breach point: open in the game (the map files hold closed ones)


# --- grid points ------------------------------------------------------------------


class _Grid:
    """Grid point [col, row] (steps of 0.5) -> board mm."""

    def __init__(self, cfg: BoardConfig, where: str):
        self.xs, self.ys = cfg.grid_lines_mm()
        self.cfg = cfg
        self.where = where

    def _axis(self, value: Any, lines: np.ndarray, what: str) -> float:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise MapError(f"{self.where}: {what} {value!r} is not a number")
        if abs(value * 2 - round(value * 2)) > 1e-9:
            raise MapError(f"{self.where}: {what} {value:g} is not on the grid or a half grid (steps of 0.5)")
        if not 0 <= value <= len(lines) - 1:
            raise MapError(f"{self.where}: {what} {value:g} is outside the grid 0..{len(lines) - 1}")
        return float(np.interp(value, np.arange(len(lines)), lines))

    def point(self, value: Any) -> tuple[float, float]:
        if not isinstance(value, list) or len(value) != 2:
            raise MapError(f"{self.where}: a grid point must be [col, row], not {value!r}")
        return self._axis(value[0], self.xs, "col"), self._axis(value[1], self.ys, "row")

    def mm(self, value: Any) -> tuple[float, float]:
        if not isinstance(value, list) or len(value) != 2 or not all(isinstance(v, int | float) and not isinstance(v, bool) for v in value):
            raise MapError(f"{self.where}: at_mm must be [x, y] in board mm, not {value!r}")
        x, y = float(value[0]), float(value[1])
        if not (0 <= x <= self.cfg.width_mm and 0 <= y <= self.cfg.height_mm):
            raise MapError(f"{self.where}: at_mm [{x:g}, {y:g}] is outside the board")
        return x, y

    def located(self, entry: Any) -> tuple[float, float]:
        """The point of a table with `at` (grid) or `at_mm` (board mm)."""
        if not isinstance(entry, dict) or ("at" in entry) == ("at_mm" in entry):
            raise MapError(f"{self.where}: give either at = [col, row] or at_mm = [x, y]")
        return self.point(entry["at"]) if "at" in entry else self.mm(entry["at_mm"])


def _key(p: tuple[float, float]) -> tuple[float, float]:
    return round(p[0], 3), round(p[1], 3)


# --- load -------------------------------------------------------------------------


def load_map(path: str | Path, cfg: BoardConfig) -> TerrainMap:
    """Read and check one map file. Raises MapError."""
    path = Path(path)
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError:
        raise MapError(f"{path}: no such file") from None
    except (OSError, ValueError) as exc:
        raise MapError(f"{path.name}: {exc}") from None
    return parse_map(raw, cfg, path)


def parse_map(raw: dict[str, Any], cfg: BoardConfig, path: Path | None = None) -> TerrainMap:
    """Check the TOML content of a map file and turn it into board mm. Raises MapError."""
    name = path.name if path is not None else "map"
    if raw.get("schema") != SCHEMA:
        raise MapError(f"{name}: schema must be {SCHEMA}, not {raw.get('schema')!r}")
    for key in ("name", "terrain"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise MapError(f"{name}: {key} must be text")
    known = {"schema", "name", "terrain", "source", "pillars", "wall_ends", "wall", "objective", "teleporter", "zone", "light"}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise MapError(f"{name}: unknown keys {', '.join(unknown)}")

    grid = _Grid(cfg, name)
    pitch = (float(np.diff(grid.xs).mean()), float(np.diff(grid.ys).mean()))

    pillars: list[tuple[float, float]] = []
    for value in _list(raw, "pillars", name):
        grid.where = f"{name}: pillar {value!r}"
        p = grid.point(value)
        if _key(p) in {_key(q) for q in pillars}:
            raise MapError(f"{grid.where}: two pillars at the same point")
        pillars.append(p)
    end_points: list[tuple[float, float]] = []
    for value in _list(raw, "wall_ends", name):
        grid.where = f"{name}: wall end {value!r}"
        p = grid.point(value)
        if _key(p) in {_key(q) for q in pillars + end_points}:
            raise MapError(f"{grid.where}: a pillar or wall end is already at this point")
        end_points.append(p)

    walls: list[Wall] = []
    for k, entry in enumerate(_list(raw, "wall", name)):
        grid.where = f"{name}: wall {k + 1}"
        if not isinstance(entry, dict):
            raise MapError(f"{grid.where}: must be a table")
        a, b = grid.point(entry.get("from")), grid.point(entry.get("to"))
        dx, dy = abs(b[0] - a[0]) / pitch[0], abs(b[1] - a[1]) / pitch[1]
        if min(dx, dy) > 1e-6:
            raise MapError(f"{grid.where}: must be horizontal or vertical")
        grids = max(dx, dy)
        key = entry.get("key")
        if not isinstance(key, str) or not WALL_KEY.match(key):
            raise MapError(f"{grid.where}: key must be the wall label of the card, for example \"A1\" or \"B3\", not {key!r}")
        grid.where = f"{name}: wall {k + 1} ({key})"
        wanted = 2 if key[0] == "A" else 1
        if abs(grids - wanted) > 1e-6:
            raise MapError(f"{grid.where}: an {key[0]} wall is {wanted} grid long, not {grids:g}")
        opening = entry.get("opening", False)
        if not isinstance(opening, bool):
            raise MapError(f"{grid.where}: opening must be true (hexagon marker next to the wall) or false, not {opening!r}")
        extra = sorted(set(entry) - {"key", "from", "to", "opening", "opening_at"})
        if extra:
            raise MapError(f"{grid.where}: unknown keys {', '.join(extra)}")
        if not opening and "opening_at" in entry:
            raise MapError(f"{grid.where}: opening_at needs opening = true")
        at = grid.point(entry["opening_at"]) if "opening_at" in entry else None
        kind = ("breach" if key in cfg.terrain_breach_walls else "hatchway") if opening else "none"
        walls.append(Wall(a, b, wanted, kind, at, key))

    seen: dict[tuple[tuple[float, float], tuple[float, float]], int] = {}
    for k, w in enumerate(walls):
        pair = tuple(sorted((_key(w.start), _key(w.end))))
        if pair in seen:
            raise MapError(f"{name}: walls {seen[pair] + 1} and {k + 1} are the same wall")
        seen[pair] = k  # type: ignore[index]

    pillar_keys = {_key(p) for p in pillars}
    end_keys = {_key(p) for p in end_points}
    for k, w in enumerate(walls):
        for p in (w.start, w.end):
            if _key(p) not in pillar_keys | end_keys:
                raise MapError(f"{name}: wall {k + 1} ends at {_grid_text(p, grid)} with no pillar or wall end")
    for k, w in enumerate(walls):
        walls[k] = _place_opening(w, pillars, f"{name}: wall {k + 1} ({w.key})", grid, cfg)
    wall_ends = []
    for p in end_points:
        touching = [w for w in walls if _key(p) in (_key(w.start), _key(w.end))]
        if len(touching) != 1:
            raise MapError(f"{name}: wall end at {_grid_text(p, grid)} must touch exactly 1 wall, not {len(touching)}")
        w = touching[0]
        other = w.end if _key(w.start) == _key(p) else w.start
        v = np.subtract(other, p)
        v = v / np.linalg.norm(v)
        wall_ends.append(WallEnd(p, (float(v[0]), float(v[1]))))

    objectives, teleporters = [], []
    for key, out in (("objective", objectives), ("teleporter", teleporters)):
        for k, entry in enumerate(_list(raw, key, name)):
            grid.where = f"{name}: {key} {k + 1}"
            out.append(grid.located(entry))

    zones = []
    for k, entry in enumerate(_list(raw, "zone", name)):
        grid.where = f"{name}: zone {k + 1}"
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
            raise MapError(f"{grid.where}: needs name, from and to")
        a, b = grid.point(entry.get("from")), grid.point(entry.get("to"))
        if not (a[0] < b[0] and a[1] < b[1]):
            raise MapError(f"{grid.where}: from must be the top-left and to the bottom-right corner")
        zones.append(Zone(entry["name"], _to_edge(a, b, grid, cfg)))

    light = []
    for k, entry in enumerate(_list(raw, "light", name)):
        grid.where = f"{name}: light {k + 1}"
        if not isinstance(entry, dict):
            raise MapError(f"{grid.where}: must be a table")
        key = entry.get("key")
        size = cfg.light_size(key) if isinstance(key, str) else None
        if size is None:
            known_keys = ", ".join(k for k, *_ in cfg.terrain_light_mm)
            raise MapError(f"{grid.where}: key must be a light terrain label of [terrain] light_mm ({known_keys}), not {key!r}")
        grid.where = f"{name}: light {k + 1} ({key})"
        extra = sorted(set(entry) - {"key", "at", "at_mm", "long"})
        if extra:
            raise MapError(f"{grid.where}: unknown keys {', '.join(extra)}")
        long_axis = entry.get("long", "x")
        if long_axis not in ("x", "y"):
            raise MapError(f"{grid.where}: long must be \"x\" or \"y\", not {long_axis!r}")
        at = grid.located(entry)
        hx, hy = _light_half(size, long_axis)
        tol = LIGHT_EDGE_TOL_MM
        if at[0] - hx < -tol or at[1] - hy < -tol or at[0] + hx > cfg.width_mm + tol or at[1] + hy > cfg.height_mm + tol:
            raise MapError(f"{grid.where}: the footprint goes over the board edge")
        light.append(Light(key, at, long_axis))

    return TerrainMap(
        name=raw["name"].strip(),
        terrain=raw["terrain"].strip(),
        source=str(raw.get("source", "")),
        path=path,
        walls=tuple(walls),
        pillars=tuple(pillars),
        wall_ends=tuple(wall_ends),
        objectives=tuple(objectives),
        teleporters=tuple(teleporters),
        zones=tuple(zones),
        light=tuple(light),
    )


def _on_wall(w: Wall, p) -> float | None:
    """Position of `p` along the wall (0 = start, 1 = end), or None when it is off the wall line."""
    a, b, q = np.asarray(w.start), np.asarray(w.end), np.asarray(p)
    d = b - a
    t = float(np.dot(q - a, d) / np.dot(d, d))
    if np.linalg.norm(a + t * d - q) > 0.01 or not 0 <= t <= 1:
        return None
    return t


def _place_opening(w: Wall, pillars, where: str, grid: _Grid, cfg: BoardConfig) -> Wall:
    """An opening sits at the centre between two pillars. A pillar inside
    the wall (the middle pillar of an A wall) splits it in two parts; then
    `opening_at` names the part. Without a middle pillar the opening is at the
    wall middle."""
    inner = sorted(t for t in (_on_wall(w, p) for p in pillars) if t is not None and 0.01 < t < 0.99)
    if inner and any(abs(t - 0.5) > 0.01 for t in inner):
        raise MapError(f"{where}: a pillar inside a wall must be at the wall middle")
    middle = w.key.startswith("A") and w.key not in cfg.terrain_no_middle_pillar
    if middle and not inner:
        raise MapError(f"{where}: an {w.key} wall has a pillar at its middle {_grid_text(_mid(w), grid)}: add it to pillars")
    if not middle and inner:
        raise MapError(f"{where}: a {w.key} wall has no pillar at its middle {_grid_text(_mid(w), grid)}: remove it from pillars")
    if w.opening == "none":
        return w
    a, b = np.asarray(w.start), np.asarray(w.end)
    cuts = [0.0, *inner, 1.0]
    centres = [0.5 * (t0 + t1) for t0, t1 in zip(cuts, cuts[1:])]
    if w.opening_at is None:
        if len(centres) > 1:
            raise MapError(f"{where}: the wall has a middle pillar, so give opening_at = the centre of the part with the opening")
        t = centres[0]
    else:
        t = _on_wall(w, w.opening_at)
        if t is None or min(abs(t - c) for c in centres) > 0.01:
            spots = ", ".join(_grid_text(tuple(a + c * (b - a)), grid) for c in centres)
            raise MapError(f"{where}: opening_at {_grid_text(w.opening_at, grid)} must be the centre between two pillars: {spots}")
    p = a + t * (b - a)
    return Wall(w.start, w.end, w.grids, w.opening, (float(p[0]), float(p[1])), w.key)


def _mid(w: Wall) -> tuple[float, float]:
    return 0.5 * (w.start[0] + w.end[0]), 0.5 * (w.start[1] + w.end[1])


def _list(raw: dict[str, Any], key: str, name: str) -> list:
    value = raw.get(key, [])
    if not isinstance(value, list):
        raise MapError(f"{name}: {key} must be a list")
    return value


def _grid_text(p: tuple[float, float], grid: _Grid) -> str:
    col = np.interp(p[0], grid.xs, np.arange(len(grid.xs)))
    row = np.interp(p[1], grid.ys, np.arange(len(grid.ys)))
    return f"[{col:g}, {row:g}]"


def _to_edge(a, b, grid: _Grid, cfg: BoardConfig) -> tuple[float, float, float, float]:
    """A zone side on an outer grid line goes on to the board edge."""
    x0 = 0.0 if np.isclose(a[0], grid.xs[0]) else a[0]
    y0 = 0.0 if np.isclose(a[1], grid.ys[0]) else a[1]
    x1 = cfg.width_mm if np.isclose(b[0], grid.xs[-1]) else b[0]
    y1 = cfg.height_mm if np.isclose(b[1], grid.ys[-1]) else b[1]
    return float(x0), float(y0), float(x1), float(y1)


# --- piece files ------------------------------------------------------------------


def piece_frame(outline) -> np.ndarray:
    """Move an outline (N, 2) into the piece frame: origin at the centre of
    its bounding box. The caller turns the long side onto x before."""
    pts = np.asarray(outline, float).reshape(-1, 2)
    return pts - 0.5 * (pts.min(axis=0) + pts.max(axis=0))


def parse_piece(raw: dict[str, Any], name: str = "piece") -> Piece:
    """Check the TOML content of a piece file. Raises MapError."""
    if raw.get("schema") != PIECE_SCHEMA:
        raise MapError(f"{name}: schema must be {PIECE_SCHEMA}, not {raw.get('schema')!r}")
    unknown = sorted(set(raw) - {"schema", "key", "outline_mm", "source"})
    if unknown:
        raise MapError(f"{name}: unknown keys {', '.join(unknown)}")
    key = raw.get("key")
    if not isinstance(key, str) or not key.strip():
        raise MapError(f"{name}: key must be the light terrain label, for example \"C1\"")
    pts = raw.get("outline_mm")
    ok = isinstance(pts, list) and len(pts) >= 3 and all(
        isinstance(p, list) and len(p) == 2 and all(isinstance(v, int | float) and not isinstance(v, bool) for v in p) for p in pts
    )
    if not ok:
        raise MapError(f"{name}: outline_mm must be a list of 3 or more points [x, y] in mm")
    outline = np.asarray(pts, float)
    x, y = outline[:, 0], outline[:, 1]
    if abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) < 2.0:  # twice the area
        raise MapError(f"{name}: outline_mm has no area")
    return Piece(key.strip(), piece_frame(outline))


def load_pieces(folder: str | Path) -> tuple[dict[str, Piece], list[str]]:
    """The piece files of one folder (`<folder>/*.toml`) by key, and a
    problem text for each file that cannot be used. No folder: no pieces."""
    pieces: dict[str, Piece] = {}
    problems: list[str] = []
    for path in sorted(Path(folder).glob("*.toml")):
        try:
            with path.open("rb") as fh:
                piece = parse_piece(tomllib.load(fh), path.name)
        except MapError as exc:
            problems.append(str(exc))
        except (OSError, ValueError) as exc:
            problems.append(f"{path.name}: {exc}")
        else:
            pieces[piece.key] = piece
    return pieces, problems


def pieces_for(tmap: TerrainMap, root: str | Path = DEFAULT_PIECES_ROOT) -> tuple[dict[str, Piece], list[str]]:
    """The piece files of the terrain type of a map: the folder under
    `root` with the name of the folder of the map file."""
    if tmap.path is None:
        return {}, []
    return load_pieces(Path(root) / tmap.path.parent.name)


def write_piece(path: str | Path, piece: Piece, source: str = "") -> None:
    """Write a piece file (skill `record-terrain`)."""
    points = ", ".join(f"[{x:.1f}, {y:.1f}]" for x, y in piece.outline)
    lines = [
        "# Light terrain piece: real outline where it touches the board (skill record-terrain).",
        "# Piece frame: origin = centre of the bounding box, x = long side. All mm.",
        f"schema = {PIECE_SCHEMA}",
        f'key = "{piece.key}"',
        *([f'source = "{source}"'] if source else []),
        f"outline_mm = [{points}]",
    ]
    Path(path).write_text("\n".join(lines) + "\n")


# --- find map files ---------------------------------------------------------------


@dataclass(frozen=True)
class MapEntry:
    terrain: str
    name: str
    path: Path


def list_maps(root: str | Path = DEFAULT_MAPS_ROOT) -> tuple[list[MapEntry], list[str]]:
    """All map files under `root` (`<root>/<folder>/*.toml`), sorted by terrain
    type and name, and a problem text for each file that cannot be read. Only
    `name` and `terrain` are read here; `load_map` checks the rest."""
    entries, problems = [], []
    for path in sorted(Path(root).glob("*/*.toml")):
        try:
            with path.open("rb") as fh:
                raw = tomllib.load(fh)
            name, terrain = raw["name"], raw["terrain"]
            if not isinstance(name, str) or not isinstance(terrain, str):
                raise ValueError("name and terrain must be text")
        except (OSError, ValueError, KeyError) as exc:
            problems.append(f"{path.name}: {exc}")
            continue
        entries.append(MapEntry(terrain.strip(), name.strip(), path))
    entries.sort(key=lambda e: (e.terrain, e.name))
    return entries, problems


# --- shapes -----------------------------------------------------------------------


def _rect(center, along, length: float, depth: float) -> np.ndarray:
    """Rectangle around `center`, `length` along the unit vector `along`, `depth` across it."""
    c, u = np.asarray(center, float), np.asarray(along, float)
    n = np.array([-u[1], u[0]])
    hl, hd = 0.5 * length * u, 0.5 * depth * n
    return np.array([c - hl - hd, c + hl - hd, c + hl + hd, c - hl + hd])


def _disc(center, diameter: float, kind: str) -> Shape:
    a = np.linspace(0, 2 * np.pi, max(48, int(np.pi * diameter / 3)), endpoint=False)
    c = np.asarray(center, float)
    return Shape(kind, c + 0.5 * diameter * np.c_[np.cos(a), np.sin(a)], (float(c[0]), float(c[1])), 0.5 * diameter)


def _light_half(size: tuple[float, float, float], long_axis: str) -> tuple[float, float]:
    long, short, _ = size
    return (0.5 * long, 0.5 * short) if long_axis == "x" else (0.5 * short, 0.5 * long)


def _light_outline(item: Light, cfg: BoardConfig) -> np.ndarray:
    """Rectangle (or octagon with a corner cut) of the card symbol, in board mm."""
    size = cfg.light_size(item.key)
    assert size is not None
    hx, hy = _light_half(size, item.long_axis)
    c = size[2]
    x, y = item.at
    if c <= 0:
        corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    else:
        corners = [(-hx + c, -hy), (hx - c, -hy), (hx, -hy + c), (hx, hy - c), (hx - c, hy), (-hx + c, hy), (-hx, hy - c), (-hx, -hy + c)]
    return np.array([(x + dx, y + dy) for dx, dy in corners], float)


def light_rect(item: Light, cfg: BoardConfig, margin_mm: float = 0.0) -> tuple[float, float, float, float]:
    """Rectangle x0, y0, x1, y1 of the card symbol of a light terrain item, grown by `margin_mm`."""
    size = cfg.light_size(item.key)
    assert size is not None
    hx, hy = _light_half(size, item.long_axis)
    return item.at[0] - hx - margin_mm, item.at[1] - hy - margin_mm, item.at[0] + hx + margin_mm, item.at[1] + hy + margin_mm


def light_outline(item: Light, cfg: BoardConfig, piece: Piece | None = None, pose: tuple[Any, float] | None = None) -> np.ndarray:
    """Footprint (N, 2) of one light terrain item in board mm: the real
    outline of the piece file at `pose` = (centre, angle in degrees), found
    on the board. Without a piece file or a pose: the card symbol at the
    map position."""
    if piece is None or pose is None:
        return _light_outline(item, cfg)
    return piece.at(*pose)


def shapes(tmap: TerrainMap, cfg: BoardConfig, pieces: dict[str, Piece] | None = None, poses: dict[int, tuple[Any, float]] | None = None) -> list[Shape]:
    """All footprints in board mm, in drawing order (zones first, discs last).
    `pieces`: piece files by key; `poses`: (centre, angle) of light item k,
    found on the board. A light item with both gets its real outline
    (`located`), else the card symbol."""
    pieces, poses = pieces or {}, poses or {}
    out: list[Shape] = []
    for z in tmap.zones:
        x0, y0, x1, y1 = z.rect
        out.append(Shape("zone", np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float), name=z.name))
    centres = []
    for k, item in enumerate(tmap.light):  # on the floor, under walls and markers
        piece, pose = pieces.get(item.key), poses.get(k)
        at = item.at if piece is None or pose is None else (float(pose[0][0]), float(pose[0][1]))
        centres.append(at)
        out.append(Shape("light", light_outline(item, cfg, piece, pose), at, name=item.key, located=piece is not None and pose is not None))
    for w in tmap.walls:
        a, b = np.asarray(w.start), np.asarray(w.end)
        length = float(np.linalg.norm(b - a))
        u = (b - a) / length
        out.append(Shape("wall", _rect(0.5 * (a + b), u, length, cfg.terrain_wall_mm)))
        mid = w.opening_at
        if w.opening == "hatchway":
            out.append(Shape("hatchway", _rect(mid, u, cfg.terrain_hatchway_width_mm, cfg.terrain_hatchway_depth_mm)))
        elif w.opening == "breach":
            width = cfg.terrain_breach_long_mm if w.grids == 2 else cfg.terrain_breach_short_mm
            out.append(Shape("breach", _rect(mid, u, width, cfg.terrain_wall_mm)))
    for p in tmap.pillars:
        out.append(Shape("pillar", _rect(p, (1.0, 0.0), cfg.terrain_pillar_mm, cfg.terrain_pillar_mm)))
    for e in tmap.wall_ends:
        half = cfg.terrain_wall_end_mm
        center = np.asarray(e.at) - 0.5 * half * np.asarray(e.toward)  # outside: it adds to the wall length
        out.append(Shape("wall_end", _rect(center, e.toward, half, cfg.terrain_pillar_mm)))
    out += [_disc(p, cfg.terrain_teleporter_mm, "teleporter") for p in tmap.teleporters]
    for p in tmap.objectives:
        out.append(_disc(p, cfg.terrain_objective_mm + 2 * cfg.terrain_objective_range_mm, "objective_range"))
        out.append(_disc(p, cfg.terrain_objective_mm, "objective"))
    # Keys last, so they are drawn on top of all other terrain.
    gap = 0.5 * cfg.terrain_pillar_mm + 3.0  # clear of the middle pillar
    for w in tmap.walls:
        c = _mid(w)
        vertical = abs(w.end[0] - w.start[0]) < 1e-6
        anchor = (c[0] + gap, c[1]) if vertical else (c[0], c[1] - gap)  # right of a vertical wall, above a horizontal one
        out.append(Shape("wall_key", np.array([anchor]), c, name=w.key, vertical=vertical))
    out += [Shape("teleporter_key", np.array([p]), p, name="T") for p in tmap.teleporters]
    out += [Shape("light_key", np.array([at]), at, name=item.key) for item, at in zip(tmap.light, centres)]
    return out


def summary(tmap: TerrainMap) -> str:
    openings = [w.opening for w in tmap.walls]
    return (
        f"{tmap.terrain} / {tmap.name}: {len(tmap.walls)} walls ({openings.count('hatchway')} hatchways, "
        f"{openings.count('breach')} breach points), {len(tmap.pillars)} pillars, {len(tmap.wall_ends)} wall ends, "
        f"{len(tmap.objectives)} objectives, {len(tmap.teleporters)} teleporters, {len(tmap.zones)} zones, "
        f"{len(tmap.light)} light terrain"
    )
