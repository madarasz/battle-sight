"""Drawing helpers. All functions draw in place on a BGR frame."""

from __future__ import annotations

import cv2
import numpy as np

from .calib import DriftReport, DriftStatus, TagObs
from .config import CORNER_NAMES, BoardConfig

GREEN = (80, 220, 80)
YELLOW = (40, 200, 255)
RED = (60, 60, 230)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
CYAN = (255, 220, 60)
MAGENTA = (230, 70, 230)
BLUE = (230, 120, 40)
GREY = (170, 170, 170)

_STATUS_COLOR = {
    DriftStatus.NO_REFERENCE: YELLOW,
    DriftStatus.OK: GREEN,
    DriftStatus.MOVED: YELLOW,
    DriftStatus.CARD_MOVED: RED,
    DriftStatus.LOST: RED,
}


def _scale(frame: np.ndarray) -> float:
    """Line and font scale relative to a 1080p frame."""
    return max(0.6, frame.shape[1] / 1920.0)


def draw_tags(frame: np.ndarray, obs: dict[int, TagObs], cfg: BoardConfig, report: DriftReport | None = None) -> None:
    s = _scale(frame)
    thick = max(1, round(2 * s))
    for tag_id, tag in obs.items():
        color = GREEN
        if report is not None and tag_id in report.moved:
            color = YELLOW
        pts = tag.corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, color, thick, cv2.LINE_AA)
        c = tuple(int(v) for v in tag.center)
        k = cfg.corner_index(tag_id)
        label = f"{tag_id}" if k is None else f"{tag_id} {CORNER_NAMES[k]}"
        cv2.putText(frame, label, (c[0] + int(8 * s), c[1] - int(8 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, BLACK, thick + 2, cv2.LINE_AA)
        cv2.putText(frame, label, (c[0] + int(8 * s), c[1] - int(8 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, color, thick, cv2.LINE_AA)
        # Tag corner 0 (its own top-left) in a different colour: shows tag orientation.
        cv2.circle(frame, tuple(int(v) for v in tag.corners[0]), max(2, int(4 * s)), CYAN, -1, cv2.LINE_AA)
    if report is not None:
        for tag_id in report.lost:
            # Nothing to draw at the tag itself, it is gone. Show where the reference was.
            pass


def draw_board_corners(frame: np.ndarray, corners: dict[int, np.ndarray], reference: dict[int, np.ndarray] | None = None) -> None:
    s = _scale(frame)
    thick = max(1, round(2 * s))
    r = int(10 * s)
    for k, p in corners.items():
        x, y = int(round(p[0])), int(round(p[1]))
        cv2.line(frame, (x - r, y), (x + r, y), WHITE, thick, cv2.LINE_AA)
        cv2.line(frame, (x, y - r), (x, y + r), WHITE, thick, cv2.LINE_AA)
        cv2.circle(frame, (x, y), r, WHITE, 1, cv2.LINE_AA)
    if len(corners) == 4:
        quad = np.array([corners[k] for k in range(4)], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [quad], True, GREEN, 1, cv2.LINE_AA)
    if reference:
        for k, p in reference.items():
            if k in corners:
                continue
            x, y = int(round(p[0])), int(round(p[1]))
            cv2.circle(frame, (x, y), r, RED, thick, cv2.LINE_AA)
            cv2.putText(frame, "lost", (x + r, y + r), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, RED, thick, cv2.LINE_AA)


def draw_banner(frame: np.ndarray, report: DriftReport, extra: str = "") -> None:
    draw_text_banner(frame, report.summary(), _STATUS_COLOR[report.status], extra)


def draw_text_banner(frame: np.ndarray, text: str, color: tuple[int, int, int], extra: str = "", row: int = 0) -> None:
    """Status strip across the top. `row` 1 puts it under the row 0 strip (for a warning)."""
    s = _scale(frame)
    h = int(44 * s)
    y0 = row * h
    cv2.rectangle(frame, (0, y0), (frame.shape[1], y0 + h), BLACK, -1)
    cv2.rectangle(frame, (0, y0), (int(12 * s), y0 + h), color, -1)
    cv2.putText(frame, text, (int(24 * s), y0 + int(30 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, color, max(1, round(2 * s)), cv2.LINE_AA)
    if extra:
        (tw, _), _ = cv2.getTextSize(extra, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, 1)
        cv2.putText(frame, extra, (frame.shape[1] - tw - int(16 * s), y0 + int(30 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, WHITE, 1, cv2.LINE_AA)


def draw_crosshair(frame: np.ndarray, at: tuple[float, float], color: tuple[int, int, int] = YELLOW) -> None:
    """A "+" at the pixel `at` (a pointer for "click the centre"), with a dark
    outline and a gap at the centre, so the point under it stays visible."""
    s = _scale(frame)
    x, y = int(round(at[0])), int(round(at[1]))
    arm, gap = int(22 * s), int(5 * s)
    for c, w in ((BLACK, max(3, round(4 * s))), (color, max(1, round(2 * s)))):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cv2.line(frame, (x + dx * gap, y + dy * gap), (x + dx * arm, y + dy * arm), c, w, cv2.LINE_AA)


def fit_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    if frame.shape[1] <= width:
        return frame
    scale = width / frame.shape[1]
    return cv2.resize(frame, (width, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)


def grid_polylines_mm(cfg: BoardConfig, step_mm: float = 10.0) -> list[np.ndarray]:
    """Grid lines and board outline as point lists in board mm. Each line is
    sampled every `step_mm`, so it bends with lens distortion when drawn in the
    camera frame."""
    xs, ys = cfg.grid_lines_mm()
    m, W, H = cfg.grid_margin_mm, cfg.width_mm, cfg.height_mm

    def seg(a, b):
        n = max(2, int(np.ceil(np.linalg.norm(np.subtract(b, a)) / step_mm)) + 1)
        return np.linspace(a, b, n)

    lines = [seg((x, m), (x, H - m)) for x in xs] + [seg((m, y), (W - m, y)) for y in ys]
    return lines


def board_outline_mm(cfg: BoardConfig, step_mm: float = 10.0) -> np.ndarray:
    c = cfg.board_corners_mm()
    pts = []
    for k in range(4):
        a, b = c[k], c[(k + 1) % 4]
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / step_mm)))
        pts.append(np.linspace(a, b, n, endpoint=False))
    return np.concatenate(pts)


def draw_grid(frame: np.ndarray, cfg: BoardConfig, to_px, color: tuple[int, int, int] = YELLOW, outline: tuple[int, int, int] | None = CYAN) -> None:
    """Draw the board grid (and outline) with `to_px`: board mm (N, 2) -> pixels (N, 2)."""
    s = _scale(frame)
    thick = max(1, round(1.5 * s))
    for line in grid_polylines_mm(cfg):
        pts = np.rint(to_px(line)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], False, color, thick, cv2.LINE_AA)
    if outline is not None:
        pts = np.rint(to_px(board_outline_mm(cfg))).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, outline, thick, cv2.LINE_AA)


def draw_bases(frame: np.ndarray, bases, to_px, color: tuple[int, int, int] = GREEN) -> None:
    """Draw found bases (circles.Circle) with `to_px`: board mm (N, 2) -> pixels (N, 2).
    The outline is the base bottom. The label is the diameter class in mm."""
    s = _scale(frame)
    thick = max(1, round(2 * s))
    a = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    for c in bases:
        r = 0.5 * c.diameter_mm
        ring = c.center_mm + r * np.c_[np.cos(a), np.sin(a)]
        pts = np.rint(to_px(ring)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, color, thick, cv2.LINE_AA)
        center = np.rint(to_px(c.center_mm.reshape(1, 2))[0]).astype(int)
        cv2.drawMarker(frame, (int(center[0]), int(center[1])), color, cv2.MARKER_CROSS, int(8 * s), thick, cv2.LINE_AA)
        top = np.rint(to_px((c.center_mm + (0.0, -r - 2.0)).reshape(1, 2))[0]).astype(int)
        label = f"{c.diameter_mm:g}"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, thick)
        org = (int(top[0] - tw / 2), int(top[1]))
        cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, BLACK, thick + 2, cv2.LINE_AA)
        cv2.putText(frame, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, color, thick, cv2.LINE_AA)


def draw_base_discs(frame: np.ndarray, bases, colors, to_px) -> None:
    """A full disc for each base (circles.Circle), one colour per base (the minimap)."""
    a = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    for c, color in zip(bases, colors):
        ring = c.center_mm + 0.5 * c.diameter_mm * np.c_[np.cos(a), np.sin(a)]
        pts = np.rint(to_px(ring)).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(frame, [pts], color, cv2.LINE_AA)


LABEL_BOX_ALPHA = 0.65  # dark box behind a name


def _label(frame: np.ndarray, text: str, org: tuple[int, int], color, s: float, thick: int, center: bool = True, box: bool = False, middle: bool = False, size: float = 1.0) -> None:
    """Text with a black outline. `org` is the top of the text, or its
    middle with `middle`. `box`: a dark box behind it, for names on a busy board.
    `size`: font size factor."""
    font = 0.6 * s * size
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, thick)
    x = int(org[0] - tw / 2) if center else int(org[0])
    y = int(org[1] + th / 2) if middle else int(org[1] + th)
    if box:
        pad = max(2, round(4 * s))
        h, w = frame.shape[:2]
        x0, y0 = max(0, x - pad), max(0, y - th - pad)
        x1, y1 = min(w, x + tw + pad), min(h, y + base + pad)
        if x1 > x0 and y1 > y0:
            area = frame[y0:y1, x0:x1]
            cv2.addWeighted(area, 1 - LABEL_BOX_ALPHA, np.zeros_like(area), LABEL_BOX_ALPHA, 0, dst=area)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font, BLACK, thick + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font, color, thick, cv2.LINE_AA)


INSIDE_NAME_SIZE = 1.6  # font size factor of the names on the minimap


def _shown_name(label: str) -> str:
    """Operative name of a label for the board: "Boy 2" for the copy label
    `team/Boy#2` (see `ident.display_name`)."""
    name = label.partition("/")[2] or label
    base, sep, k = name.partition("#")
    return f"{base} {k}" if sep else name


def ident_color(x) -> tuple[int, int, int]:
    """Green: sure or confirmed name. Yellow: unsure name. Red: unknown."""
    if x.label is None:
        return RED
    return GREEN if x.sure else YELLOW


def draw_idents(frame: np.ndarray, bases, idents, to_px, scores: bool = True, team_colors: dict[str, tuple[int, int, int]] | None = None, inside: bool = False) -> None:
    """Operative names (ident.Ident) under the bases (circles.Circle), on a
    dark box; with `inside` (minimap) in the middle of the base, in a larger
    font (`INSIDE_NAME_SIZE`). The team part of the label is left out. `scores`: the score
    after a name that is not confirmed. `team_colors`: circle and name colour
    per team; then a base without a name gets a grey circle and `?`. Without
    it the colour is `ident_color`."""
    s = _scale(frame)
    thick = max(1, round(2 * s))
    a = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    for c, x in zip(bases, idents):
        r = 0.5 * c.diameter_mm
        color = ident_color(x)
        ring_color = color
        if team_colors:
            team = None if x.label is None else x.label.partition("/")[0]
            ring_color = team_colors.get(team, GREY if team is None else color)
        ring = c.center_mm + r * np.c_[np.cos(a), np.sin(a)]
        pts = np.rint(to_px(ring)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, ring_color, thick, cv2.LINE_AA)
        name = "?" if x.label is None else _shown_name(x.label)
        if scores and x.label is not None and not x.confirmed:
            name += f" {x.score:.2f}"
        at = c.center_mm if inside else c.center_mm + (0.0, r + 2.0)
        org = np.rint(to_px(np.asarray(at, float).reshape(1, 2))[0]).astype(int)
        size = INSIDE_NAME_SIZE if inside else 1.0
        _label(frame, name, (int(org[0]), int(org[1])), ring_color, s, max(thick, round(thick * size)), box=True, middle=inside, size=size)


def draw_outline_mm(frame: np.ndarray, points_mm, to_px, color: tuple[int, int, int]) -> None:
    """Heavy outline of a polygon in board mm (a selected terrain item)."""
    s = _scale(frame)
    thick = max(1, round(3 * s))
    pts = np.rint(to_px(_dense(np.asarray(points_mm, float), 5.0))).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], True, BLACK, thick + 2, cv2.LINE_AA)
    cv2.polylines(frame, [pts], True, color, thick, cv2.LINE_AA)


def draw_circle_mm(frame: np.ndarray, center_mm, radius_mm: float, to_px, color: tuple[int, int, int]) -> None:
    """Heavy circle in board mm, sampled about every 5 mm, so that it bends
    with lens distortion when drawn in the camera frame."""
    n = max(64, int(np.ceil(2 * np.pi * radius_mm / 5.0)))
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.asarray(center_mm, float) + radius_mm * np.c_[np.cos(a), np.sin(a)]
    pts = np.rint(to_px(ring)).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], True, color, max(1, round(4 * _scale(frame))), cv2.LINE_AA)


def draw_rim_line(frame: np.ndarray, start_mm, end_mm, text: str, to_px, color: tuple[int, int, int] = MAGENTA) -> None:
    """Line between two points in board mm, with `text` at its middle."""
    draw_path_mm(frame, [start_mm, end_mm], text, to_px, color)


def draw_path_mm(frame: np.ndarray, points_mm, text: str, to_px, color: tuple[int, int, int] = MAGENTA) -> None:
    """Path of line segments in board mm, with `text` at half its length.
    Each segment has extra points, so that it bends with lens distortion."""
    s = _scale(frame)
    thick = max(1, round(2 * s))
    pts = np.asarray(points_mm, float).reshape(-1, 2)
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    line = [np.linspace(a, b, max(2, int(np.ceil(n / 5.0)) + 1)) for a, b, n in zip(pts[:-1], pts[1:], steps)]
    px = to_px(np.concatenate(line))
    cv2.polylines(frame, [np.rint(px).astype(np.int32).reshape(-1, 1, 2)], False, color, thick, cv2.LINE_AA)
    along = np.concatenate([[0.0], np.cumsum(steps)])
    half = 0.5 * along[-1]
    k = min(int(np.searchsorted(along, half, side="right")) - 1, len(steps) - 1)
    at = pts[k] + (pts[k + 1] - pts[k]) * ((half - along[k]) / steps[k] if steps[k] > 0 else 0.0)
    middle = np.rint(to_px(at.reshape(1, 2))[0]).astype(int)
    _label(frame, text, (int(middle[0]), int(middle[1])), WHITE, s, thick)


MOVE_COLOR = (215, 215, 215)


def draw_move_mm(frame: np.ndarray, points_mm, radius_mm: float, costs, total: int, to_px, color: tuple[int, int, int] = MOVE_COLOR) -> None:
    """Movement of a base in straight increments: the base at each bend
    and at the end, the two lines that its rim follows, the inches of each
    increment at its middle (for more than one increment) and the inches
    in all below the end."""
    s = _scale(frame)
    thick = max(1, round(2 * s))
    pts = np.asarray(points_mm, float).reshape(-1, 2)

    def px(points: np.ndarray) -> np.ndarray:
        return np.rint(to_px(points)).astype(np.int32).reshape(-1, 1, 2)

    turn = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    for a, b in zip(pts[:-1], pts[1:]):
        length = float(np.linalg.norm(b - a))
        if length < 1e-6:
            continue
        side = radius_mm * np.array([-(b - a)[1], (b - a)[0]]) / length
        n = max(2, int(np.ceil(length / 5.0)) + 1)
        for off in (side, -side):
            cv2.polylines(frame, [px(np.linspace(a + off, b + off, n))], False, color, thick, cv2.LINE_AA)
    for p in pts[1:]:
        cv2.polylines(frame, [px(p + radius_mm * np.c_[np.cos(turn), np.sin(turn)])], True, color, thick, cv2.LINE_AA)
    if len(costs) > 1:
        for a, b, cost in zip(pts[:-1], pts[1:], costs):
            middle = px(0.5 * (a + b).reshape(1, 2))[0, 0]
            _label(frame, f'{cost}"', (int(middle[0]), int(middle[1])), color, s, thick)
    below = px((pts[-1] + (0.0, radius_mm + 2.0)).reshape(1, 2))[0, 0]
    _label(frame, f'{total}"', (int(below[0]), int(below[1])), WHITE, s, thick, box=True)


RANGE_ALPHA = 0.25


def draw_range_mm(frame: np.ndarray, contours_mm, to_px, color: tuple[int, int, int] = MAGENTA) -> None:
    """Area in board mm (closed contours, a contour in another one is a
    hole): a light fill and a heavy edge, as heavy as `draw_circle_mm`."""
    polys = [np.rint(to_px(_dense(np.asarray(c, float), 5.0))).astype(np.int32).reshape(-1, 1, 2) for c in contours_mm if len(c) >= 3]
    if not polys:
        return
    layer = frame.copy()
    cv2.fillPoly(layer, polys, color, cv2.LINE_AA)  # even-odd: holes stay clear
    cv2.addWeighted(layer, RANGE_ALPHA, frame, 1 - RANGE_ALPHA, 0, dst=frame)
    cv2.polylines(frame, polys, True, color, max(1, round(4 * _scale(frame))), cv2.LINE_AA)


def draw_dashed_circle(
    frame: np.ndarray,
    center: tuple[float, float],
    radius: float,
    color: tuple[int, int, int],
    thickness: int = 1,
    dashes: int = 24,
    duty: float = 0.5,
) -> None:
    """Dashed circle in pixels: `dashes` arcs, each `duty` of its step long.
    A small duty with many dashes gives a dotted circle."""
    c = (int(round(center[0])), int(round(center[1])))
    r = max(1, int(round(radius)))
    step = 360.0 / dashes
    for k in range(dashes):
        a0 = k * step
        cv2.ellipse(frame, c, (r, r), 0, a0, a0 + duty * step, color, thickness, cv2.LINE_AA)


ORANGE = (0, 140, 255)
TERRAIN_FILL = {
    "wall": (235, 235, 235),
    "pillar": (170, 170, 170),
    "wall_end": (170, 170, 170),
    "hatchway": YELLOW,
    "breach": ORANGE,
}
OPENING_KINDS = ("hatchway", "breach")
ZONE_FILL = {"orange": (40, 110, 220), "grey": (220, 220, 220)}
ZONE_ALPHA = 0.3
LIGHT_FILL = (60, 75, 90)  # light terrain (C1..C5): dark brown-grey, half see-through
LIGHT_ALPHA = 0.55


def _dense(poly: np.ndarray, step_mm: float = 10.0) -> np.ndarray:
    """Closed polygon in mm with extra points every `step_mm`, so that it bends
    with lens distortion when drawn in the camera frame."""
    pts = []
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step_mm)))
        pts.append(np.linspace(a, b, n, endpoint=False))
    return np.concatenate(pts)


def _draw_open(frame: np.ndarray, floor: np.ndarray, sh, px, thick: int) -> None:
    """An open hatchway or breach point: a gap in its wall (the floor of
    `floor` shows through) with a jamb in the opening colour at each end."""
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillPoly(mask, [px(_dense(sh.points, 5.0))], 255)
    cv2.dilate(mask, np.ones((3, 3), np.uint8), dst=mask)  # also the anti-aliased wall edge
    frame[mask > 0] = floor[mask > 0]
    pts = np.asarray(sh.points, float)
    edges = [(pts[k], pts[(k + 1) % len(pts)]) for k in range(len(pts))]
    edges.sort(key=lambda e: float(np.linalg.norm(e[1] - e[0])))
    for a, b in edges[:2]:  # the two short sides: across the wall
        seg = px(np.linspace(a, b, 8))
        cv2.polylines(frame, [seg], False, BLACK, thick + 3, cv2.LINE_AA)
        cv2.polylines(frame, [seg], False, TERRAIN_FILL[sh.kind], thick + 1, cv2.LINE_AA)


def draw_terrain(frame: np.ndarray, shapes, to_px) -> None:
    """Draw terrain footprints (terrain.Shape, board mm) with `to_px`: zones
    as a light fill, walls, pillars and closed openings filled, an open
    opening as a gap in its wall with a jamb at each end, objectives and
    teleporter pads as rings, a dashed circle 1 in around each objective,
    light terrain as a half see-through dark fill, wall keys above horizontal
    walls and right of vertical walls, `T` and `C1`.. keys at the centre."""
    s = _scale(frame)
    thick = max(1, round(2 * s))

    def px(points: np.ndarray) -> np.ndarray:
        return np.rint(to_px(points)).astype(np.int32).reshape(-1, 1, 2)

    zones = [sh for sh in shapes if sh.kind == "zone"]
    if zones:
        layer = frame.copy()
        for sh in zones:
            cv2.fillPoly(layer, [px(_dense(sh.points))], ZONE_FILL.get(sh.name, ZONE_FILL["grey"]), cv2.LINE_AA)
        cv2.addWeighted(layer, ZONE_ALPHA, frame, 1 - ZONE_ALPHA, 0, dst=frame)
        for sh in zones:
            cv2.polylines(frame, [px(_dense(sh.points))], True, ZONE_FILL.get(sh.name, ZONE_FILL["grey"]), thick, cv2.LINE_AA)
    light = [sh for sh in shapes if sh.kind == "light"]
    if light:
        layer = frame.copy()
        for sh in light:
            cv2.fillPoly(layer, [px(_dense(sh.points))], LIGHT_FILL, cv2.LINE_AA)
        cv2.addWeighted(layer, LIGHT_ALPHA, frame, 1 - LIGHT_ALPHA, 0, dst=frame)
        for sh in light:
            cv2.polylines(frame, [px(_dense(sh.points))], True, BLACK, thick, cv2.LINE_AA)
    floor = frame.copy() if any(sh.open and sh.kind in OPENING_KINDS for sh in shapes) else None
    for sh in shapes:
        if sh.open and sh.kind in OPENING_KINDS:
            _draw_open(frame, floor, sh, px, thick)
        elif sh.kind in TERRAIN_FILL:
            pts = px(_dense(sh.points, 5.0))
            cv2.fillPoly(frame, [pts], TERRAIN_FILL[sh.kind], cv2.LINE_AA)
            cv2.polylines(frame, [pts], True, BLACK, 1, cv2.LINE_AA)
        elif sh.kind == "objective":
            cv2.polylines(frame, [px(sh.points)], True, BLACK, thick + 2, cv2.LINE_AA)
            cv2.polylines(frame, [px(sh.points)], True, WHITE, thick, cv2.LINE_AA)
            c = px(np.array([sh.center]))[0, 0]
            cv2.drawMarker(frame, (int(c[0]), int(c[1])), WHITE, cv2.MARKER_TILTED_CROSS, int(10 * s), thick, cv2.LINE_AA)
        elif sh.kind == "teleporter":
            cv2.polylines(frame, [px(sh.points)], True, BLACK, thick + 2, cv2.LINE_AA)
            cv2.polylines(frame, [px(sh.points)], True, CYAN, thick, cv2.LINE_AA)
        elif sh.kind in ("wall_key", "teleporter_key", "light_key"):
            a = to_px(sh.points)[0]
            (tw, th), _ = cv2.getTextSize(sh.name, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, thick)
            if sh.kind in ("teleporter_key", "light_key"):
                org = (int(a[0] - tw / 2), int(a[1] + th / 2))  # centred on the pad or the light terrain
            else:
                org = (int(a[0]), int(a[1] + th / 2)) if sh.vertical else (int(a[0] - tw / 2), int(a[1]))
            pad = max(2, int(4 * s))
            box = frame[max(0, org[1] - th - pad) : org[1] + pad, max(0, org[0] - pad) : org[0] + tw + pad]
            box[:] = (box * 0.25).astype(frame.dtype)  # dark box under the text
            cv2.putText(frame, sh.name, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, WHITE, thick, cv2.LINE_AA)
        elif sh.kind == "objective_range":
            pts = to_px(sh.points)
            for k in range(0, len(pts), 2):  # dashes: every other segment of the sampled circle
                seg = np.rint(pts[[k, (k + 1) % len(pts)]]).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [seg], False, WHITE, thick, cv2.LINE_AA)


HEALTH_MM_PER_WOUND = 2.0  # bar length per maximum wound: 15 W = 30 mm
HEALTH_BAR_MM = 4.0  # bar height
HEALTH_GAP_MM = 3.0  # from the base rim to the bar
HEALTH_EMPTY = (60, 60, 60)
HEALTH_UNHARMED = (80, 220, 80)  # all wounds left (green)
HEALTH_WOUNDED = (60, 170, 255)  # half of the wounds or more left (orange)
HEALTH_INJURED = (60, 60, 230)  # fewer than half of the wounds left (red)


def health_color(current: int, maximum: int) -> tuple[int, int, int]:
    """Unharmed: all wounds. Wounded: half of the wounds or more. Injured:
    fewer than half of the wounds."""
    if current >= maximum:
        return HEALTH_UNHARMED
    return HEALTH_WOUNDED if current >= maximum / 2 else HEALTH_INJURED


def draw_health_bars(frame: np.ndarray, bars, to_px) -> None:
    """A health bar above each base. `bars`: (centre mm, base diameter mm,
    current wounds, maximum wounds). The bar length is in proportion to the
    maximum wounds, the filled part to the current wounds."""
    s = _scale(frame)
    thick = max(1, round(s))
    for center, diameter, current, maximum in bars:
        cx, cy = float(center[0]), float(center[1])
        length = HEALTH_MM_PER_WOUND * maximum
        x0, y1 = cx - length / 2, cy - diameter / 2 - HEALTH_GAP_MM
        y0 = y1 - HEALTH_BAR_MM

        def poly(xa: float, xb: float) -> np.ndarray:
            pts = np.array([[xa, y0], [xb, y0], [xb, y1], [xa, y1]], float)
            return np.rint(to_px(pts)).astype(np.int32).reshape(-1, 1, 2)

        cv2.fillPoly(frame, [poly(x0, x0 + length)], HEALTH_EMPTY, cv2.LINE_AA)
        part = max(0, min(current, maximum)) / maximum
        if part > 0:
            cv2.fillPoly(frame, [poly(x0, x0 + length * part)], health_color(current, maximum), cv2.LINE_AA)
        cv2.polylines(frame, [poly(x0, x0 + length)], True, HEALTH_INJURED if current <= 0 else BLACK, thick, cv2.LINE_AA)


# --- control marks ----------------------------------------------------------------

MARK_COLOR = (0, 225, 255)  # bright yellow: melee signs and objective arrows
MARK_BACK = (30, 30, 30)  # dark sign behind the crossed swords
MELEE_SIGN_MM = 13.0  # radius of the melee sign
ARROW_LEN_MM = 5.0  # depth of one arrow head
ARROW_HALF_MM = 6.0  # half width of one arrow head
ARROW_STEP_MM = 7.0  # from one arrow head to the next


def _px_per_mm(to_px, at_mm) -> float:
    """Image pixels for 1 mm of board at this board point (mean of x and y)."""
    p = np.asarray(at_mm, float)
    q = np.asarray(to_px(np.array([p, p + (1.0, 0.0), p + (0.0, 1.0)])), float)
    return 0.5 * float(np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[0]))


def draw_dashed_circle_mm(frame: np.ndarray, center_mm, radius_mm: float, to_px, color: tuple[int, int, int], dash_mm: float = 5.0) -> None:
    """Dashed circle in board mm (dash and space each about `dash_mm`),
    sampled in mm so that it bends with the lens in the camera frame."""
    dashes = max(8, int(round(np.pi * radius_mm / dash_mm)))
    c = np.asarray(center_mm, float)
    thick = max(1, round(2 * _scale(frame)))
    for k in range(dashes):
        a = (k + np.linspace(0.0, 0.5, 5)) * (2 * np.pi / dashes)
        arc = np.rint(to_px(c + radius_mm * np.c_[np.cos(a), np.sin(a)])).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [arc], False, color, thick, cv2.LINE_AA)


def draw_melee_sign(frame: np.ndarray, at_mm, to_px, color: tuple[int, int, int] = MARK_COLOR) -> None:
    """Crossed swords on a dark octagon (a stop sign), centred on a board
    point. The sign stays upright; its size follows the board scale."""
    c = np.asarray(to_px(np.asarray(at_mm, float).reshape(1, 2)), float).reshape(2)
    R = max(10.0, MELEE_SIGN_MM * _px_per_mm(to_px, at_mm))
    a = np.deg2rad(22.5 + 45.0 * np.arange(8))
    octagon = np.rint(c + R * np.c_[np.cos(a), np.sin(a)]).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(frame, [octagon], MARK_BACK, cv2.LINE_AA)
    cv2.polylines(frame, [octagon], True, color, max(1, round(0.08 * R)), cv2.LINE_AA)
    for sx in (1.0, -1.0):  # blade from the lower corner to the upper corner
        tip, pommel = c + 0.46 * R * np.array([sx, -1.0]), c + 0.46 * R * np.array([-sx, 1.0])
        along = (tip - pommel) / np.linalg.norm(tip - pommel)
        normal = np.array([-along[1], along[0]])
        span = float(np.linalg.norm(tip - pommel))
        guard = pommel + 0.27 * span * along
        w = 0.075 * R  # half width of the blade
        blade = np.array([guard + w * normal, tip - 2 * w * along + w * normal, tip, tip - 2 * w * along - w * normal, guard - w * normal])
        grip = np.array([pommel + 0.45 * w * normal, guard + 0.45 * w * normal, guard - 0.45 * w * normal, pommel - 0.45 * w * normal])
        bar = np.array([guard + 0.22 * R * normal + 0.6 * w * along, guard + 0.22 * R * normal - 0.6 * w * along, guard - 0.22 * R * normal - 0.6 * w * along, guard - 0.22 * R * normal + 0.6 * w * along])
        for poly in (blade, grip, bar):
            cv2.fillPoly(frame, [np.rint(poly).astype(np.int32).reshape(-1, 1, 2)], color, cv2.LINE_AA)
        cv2.circle(frame, tuple(int(v) for v in np.rint(pommel)), max(1, round(0.9 * w)), color, -1, cv2.LINE_AA)


def objective_arrows_mm(center_mm, diameter_mm: float, objective_mm) -> list[np.ndarray]:
    """Arrow heads (3 points each: arm, tip, arm) in board mm. From the base
    rim to the objective centre, one head each `ARROW_STEP_MM`. When the
    centre is nearer to the rim than two heads (or inside the base), four
    heads around the base, at 45 degrees, point to its rim."""
    c, o = np.asarray(center_mm, float), np.asarray(objective_mm, float)
    r = 0.5 * diameter_mm
    between = float(np.linalg.norm(o - c))

    def head(tip: np.ndarray, toward: np.ndarray) -> np.ndarray:
        side = np.array([-toward[1], toward[0]])
        back = tip - ARROW_LEN_MM * toward
        return np.array([back + ARROW_HALF_MM * side, tip, back - ARROW_HALF_MM * side])

    gap = between - r
    if gap < 2 * ARROW_STEP_MM:
        out = []
        for a in np.deg2rad([45.0, 135.0, 225.0, 315.0]):
            w = np.array([np.cos(a), np.sin(a)])
            out.append(head(c + (r + 1.0) * w, -w))
        return out
    u = (o - c) / between
    rim = c + r * u
    n = int((gap - ARROW_LEN_MM) // ARROW_STEP_MM) + 1
    return [head(rim + (ARROW_LEN_MM + k * ARROW_STEP_MM) * u, u) for k in range(n)]


def draw_objective_arrows(frame: np.ndarray, center_mm, diameter_mm: float, objective_mm, to_px) -> None:
    """Yellow arrow heads with a dark outline (`objective_arrows_mm`)."""
    heads = [np.rint(to_px(h)).astype(np.int32).reshape(-1, 1, 2) for h in objective_arrows_mm(center_mm, diameter_mm, objective_mm)]
    thick = max(2, round(1.2 * _px_per_mm(to_px, center_mm)))
    cv2.polylines(frame, heads, False, BLACK, thick + 3, cv2.LINE_AA)
    cv2.polylines(frame, heads, False, MARK_COLOR, thick, cv2.LINE_AA)
