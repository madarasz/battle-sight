"""Movement of a base around terrain, in board mm. No window, no camera.

A move is a chain of straight steps of 1 in or less, and each step counts
as 1 in. Thus a straight increment counts as its length rounded up to a
full inch. The base cannot overlap a wall, a pillar, a closed opening or an
enemy base, and it stays on the board. Through an open hatchway or breach
point only the centre of the base must go (the base size does not matter
there), and that costs 1 in more. Near an open gap the base can thus be
over the wall parts while it moves, but it cannot stop there.

`move_field` gives a `MoveField`: the number of inches to each position of
the base centre, on a raster (`MOVE_PX_PER_MM`). `MoveField.outline` is the
area that the base can cover, `MoveField.route` the increments to a point.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import cv2
import numpy as np

from .measure import MM_PER_INCH

MOVE_PX_PER_MM = 1.0  # raster of the movement field
SLACK_MM = 0.5  # the base can be this far into terrain (detection error of a base that touches a wall)
ESCAPE_MM = 3.0  # a base that starts in terrain gets out of it over this length
GAP_PAST_MM = 3.0  # the zone around an open gap, where the base can be over the wall, is this much more than the base radius
GAP_WALL_MM = 1.0  # in that zone the centre keeps this distance to the terrain
RAY_ANGLES = 240
ROUND_SLACK_IN = 0.06  # an increment this much over a full inch still counts as that inch (raster of the field)
LINE_TOLERANCE = 1  # blocked samples that a straight line of the route can have (raster edge)
OUTLINE_ZOOM = 4  # the outline comes from the mask at this zoom, for a smooth edge
UNREACHED = 255


@dataclass(frozen=True)
class _Tables:
    """Offsets for one step of 1 in: rays from a source, and the pixels of the disc with their ray."""

    half: int  # the disc fits in a square of 2 * half + 1
    ray_y: np.ndarray  # (angles, radii)
    ray_x: np.ndarray
    disc_y: np.ndarray  # (P,)
    disc_x: np.ndarray
    disc_angle: np.ndarray
    disc_radius: np.ndarray
    kernel: np.ndarray  # the disc as a morphology kernel


@lru_cache(maxsize=None)
def _tables(radius_px: float) -> _Tables:
    half = int(np.ceil(radius_px))
    radii = np.arange(int(np.floor(radius_px)) + 1)
    angles = np.arange(RAY_ANGLES) * (2 * np.pi / RAY_ANGLES)
    ray_y = np.rint(np.sin(angles)[:, None] * radii).astype(np.intp)
    ray_x = np.rint(np.cos(angles)[:, None] * radii).astype(np.intp)
    dy, dx = np.mgrid[-half : half + 1, -half : half + 1]
    dist = np.hypot(dy, dx)
    inside = dist <= radius_px
    angle = np.rint(np.arctan2(dy, dx) / (2 * np.pi / RAY_ANGLES)).astype(np.intp) % RAY_ANGLES
    radius = np.minimum(np.rint(dist).astype(np.intp), len(radii) - 1)
    return _Tables(half, ray_y, ray_x, dy[inside], dx[inside], angle[inside], radius[inside], inside.astype(np.uint8))


def _visible(free: np.ndarray, ys: np.ndarray, xs: np.ndarray, t: _Tables, escape: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Pixels within one step of the sources that a source can get to in a
    straight line through `free`. `escape`: a source in terrain gets out of
    it over this number of pixels."""
    rays = free[ys[:, None, None] + t.ray_y, xs[:, None, None] + t.ray_x]
    if escape:
        rays = rays | ((np.maximum.accumulate(rays, axis=2) == 0) & (np.arange(rays.shape[2]) <= escape))
    seen = np.minimum.accumulate(rays, axis=2)[:, t.disc_angle, t.disc_radius]
    ty, tx = ys[:, None] + t.disc_y, xs[:, None] + t.disc_x
    hit = (seen > 0) & (free[ty, tx] > 0)
    return ty[hit], tx[hit]


def _seen_from(free: np.ndarray, start: tuple[int, int], radius_px: float, escape: int) -> tuple[np.ndarray, np.ndarray]:
    """For the whole raster: the pixels that the start can get to in one
    straight line of `radius_px` or less, and the distance of each pixel."""
    h, w = free.shape
    radii = np.arange(int(np.ceil(radius_px)) + 1)
    count = max(RAY_ANGLES, int(np.ceil(2 * np.pi * radius_px * 1.5)))
    angles = np.arange(count) * (2 * np.pi / count)
    ry = start[0] + np.rint(np.sin(angles)[:, None] * radii).astype(np.intp)
    rx = start[1] + np.rint(np.cos(angles)[:, None] * radii).astype(np.intp)
    on = (ry >= 0) & (ry < h) & (rx >= 0) & (rx < w)
    rays = np.where(on, free[np.clip(ry, 0, h - 1), np.clip(rx, 0, w - 1)], 0)
    rays = rays | ((np.maximum.accumulate(rays, axis=1) == 0) & (radii <= escape))
    seen = np.minimum.accumulate(rays, axis=1)
    dy, dx = np.mgrid[0:h, 0:w]
    dy, dx = dy - start[0], dx - start[1]
    dist = np.hypot(dy, dx)
    angle = np.rint(np.arctan2(dy, dx) / (2 * np.pi / count)).astype(np.intp) % count
    radius = np.minimum(np.rint(dist).astype(np.intp), len(radii) - 1)
    return (seen[angle, radius] > 0) & (free > 0) & (dist <= radius_px), dist


def _line(a, b) -> tuple[np.ndarray, np.ndarray]:
    """Pixels on the line from a to b (y, x), about 2 for each pixel of length."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(2, int(np.ceil(np.linalg.norm(b - a))) * 2 + 1)
    pts = np.rint(a + (b - a) * np.linspace(0.0, 1.0, n)[:, None]).astype(np.intp)
    return pts[:, 0], pts[:, 1]


@dataclass
class MoveField:
    """Inches to each position of the base centre. The arrays have a blocked
    band of `pad` pixels on each side; pixel (i, j) is at `origin + (j, i) / k` mm."""

    steps: np.ndarray  # uint8 inches, UNREACHED = no way within the budget
    free: np.ndarray  # the centre can move here (with the zones around the open gaps)
    stand: np.ndarray  # the centre can stop here: the base is clear of the terrain
    barrier: np.ndarray  # the lines in the open gaps: 1 in more to go over one
    origin: np.ndarray  # board mm of pixel (0, 0)
    start: tuple[int, int]  # pixel (y, x) of the base centre
    center: np.ndarray  # board mm
    radius: float  # base radius, mm
    inches: int  # the budget
    k: float = MOVE_PX_PER_MM

    def _px(self, point_mm) -> tuple[int, int]:
        x, y = np.rint((np.asarray(point_mm, float) - self.origin) * self.k).astype(int)
        return int(y), int(x)

    def _mm(self, px) -> np.ndarray:
        return self.origin + np.array([px[1], px[0]], float) / self.k

    def _inside(self, y: int, x: int) -> bool:
        return 0 <= y < self.steps.shape[0] and 0 <= x < self.steps.shape[1]

    def cost(self, point_mm) -> int | None:
        """Inches to put the base centre on the point, None when out of reach."""
        y, x = self._px(point_mm)
        if not self._inside(y, x) or not self._reached()[y, x]:
            return None
        return int(self.steps[y, x])

    def _reached(self) -> np.ndarray:
        """The positions where the base can stop within the budget."""
        return (self.steps <= self.inches) & (self.stand > 0)

    def outline(self) -> list[np.ndarray]:
        """The area that the base can cover within the budget: closed
        contours in board mm (a contour in another one is a hole)."""
        reached = self._reached().astype(np.uint8) * 255
        r = int(round(self.radius * self.k))
        cover = cv2.dilate(reached, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        z = OUTLINE_ZOOM
        big = cv2.resize(cover, None, fx=z, fy=z, interpolation=cv2.INTER_LINEAR)
        contours, _ = cv2.findContours((big > 127).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # Pixel centre c of the zoomed image is at (c + 0.5) / z - 0.5 of the mask.
        return [self.origin + ((c.reshape(-1, 2) + 0.5) / z - 0.5) / self.k for c in contours if len(c) >= 3]

    # --- route ----------------------------------------------------------------------

    def _snap(self, point_mm) -> tuple[int, int] | None:
        """The nearest position of the centre in reach, for a point on the
        covered area (at most one base radius away)."""
        y, x = self._px(point_mm)
        r = int(np.ceil(self.radius * self.k)) + 1
        y0, y1, x0, x1 = max(y - r, 0), min(y + r + 1, self.steps.shape[0]), max(x - r, 0), min(x + r + 1, self.steps.shape[1])
        if y1 <= y0 or x1 <= x0:
            return None
        ys, xs = np.nonzero(self._reached()[y0:y1, x0:x1])
        if len(ys) == 0:
            return None
        d = np.hypot(ys + y0 - y, xs + x0 - x)
        j = int(np.argmin(d))
        return (int(ys[j] + y0), int(xs[j] + x0)) if d[j] <= self.radius * self.k + 1.0 else None

    def _segment_cost(self, a, b, from_start: bool) -> int | None:
        """Inches of one straight increment between two pixels, None when terrain is in the way."""
        ys, xs = _line(a, b)
        tolerance = LINE_TOLERANCE + (int(ESCAPE_MM * self.k) + 1 if from_start else 0)
        if np.count_nonzero(self.free[ys, xs] == 0) > tolerance:
            return None
        on = self.barrier[ys, xs] > 0
        gaps = int(on[0]) + int(np.count_nonzero(on[1:] & ~on[:-1]))
        length_in = float(np.hypot(*np.subtract(a, b))) / self.k / MM_PER_INCH
        return int(np.ceil(max(0.0, length_in - ROUND_SLACK_IN))) + gaps

    def _back(self, cur: tuple[int, int], level: int, t: _Tables) -> tuple[int, int] | None:
        """A position one step (or one step through a gap) before `cur`."""
        if level == 1:
            return self.start
        ys, xs = cur[0] + t.disc_y, cur[1] + t.disc_x
        lv = np.where(self.barrier[ys, xs] > 0, UNREACHED, self.steps[ys, xs])  # no bend on a gap line
        closed = self.free & ~self.barrier
        # The strict line check first. The rays of the field and this line
        # check can differ by a pixel at a corner: then a loose check.
        for mask, limit, through, loose in ((closed, level - 1, False, 0), (self.free, level - 2, True, 0), (closed, level - 1, False, 3), (self.free, level - 2, True, 3)):
            pick = np.flatnonzero(lv <= limit)
            if len(pick) == 0:
                continue
            n = 2 * t.half + 1
            frac = np.linspace(0.0, 1.0, n)[None, :]
            ly = np.rint(cur[0] + (ys[pick, None] - cur[0]) * frac).astype(np.intp)
            lx = np.rint(cur[1] + (xs[pick, None] - cur[1]) * frac).astype(np.intp)
            blocked = np.count_nonzero(mask[ly, lx] == 0, axis=1)
            at_start = (ys[pick] == self.start[0]) & (xs[pick] == self.start[1])
            ok = blocked <= LINE_TOLERANCE + loose + at_start * (int(ESCAPE_MM * self.k) + 1)
            if through:
                ok &= (self.barrier[ly, lx] > 0).any(axis=1)
            if not ok.any():
                continue
            full = ok & (np.hypot(ys[pick] - cur[0], xs[pick] - cur[1]) <= MM_PER_INCH * self.k)
            pick = pick[full if full.any() else ok]  # a step of 1 in or less, when there is one
            # The position nearest to the start: the chain then has few bends.
            j = pick[int(np.argmin(np.hypot(ys[pick] - self.start[0], xs[pick] - self.start[1])))]
            return int(ys[j]), int(xs[j])
        return None

    def route(self, point_mm) -> tuple[int, np.ndarray, list[int]] | None:
        """The increments that put the base on the point: (inches in all,
        positions of the base centre (K, 2) in board mm from the start,
        inches of each increment). A point on the covered area that the
        centre cannot get to goes to the nearest position in reach. None
        when the point is out of reach."""
        # A little more than one step: on the raster, the position one step
        # back can be a part of a pixel out of the disc.
        t = _tables(MM_PER_INCH * self.k + 1.5)
        end = self._px(point_mm)
        exact = self._inside(*end) and bool(self._reached()[end])
        if not exact:
            end = self._snap(point_mm)
            if end is None:
                return None
        total = int(self.steps[end])
        chain, cur = [end], end
        while cur != self.start:
            cur = self._back(cur, int(self.steps[cur]), t)
            if cur is None or len(chain) > self.inches + 2:
                return None
            chain.append(cur)
        chain.reverse()
        levels = [int(self.steps[p]) for p in chain]
        # Straight increments: the positions of the chain that give the
        # fewest inches, then the fewest increments.
        n = len(chain)
        best: list[tuple[int, int, int]] = [(0, 0, -1)] + [(10**6, 0, -1)] * (n - 1)  # inches, increments, position before
        seg: dict[tuple[int, int], int] = {}
        for j in range(1, n):
            for i in range(j):
                cost = self._segment_cost(chain[i], chain[j], i == 0)
                if cost is None or (j == i + 1 and cost > levels[j] - levels[i]):
                    if j != i + 1:
                        continue
                    cost = levels[j] - levels[i]  # one step of the field: trust the field
                seg[i, j] = cost
                best[j] = min(best[j], (best[i][0] + cost, best[i][1] + 1, i))
        kept, costs, j = [n - 1], [], n - 1
        while j > 0:
            i = best[j][2]
            costs.append(seg[i, j])
            kept.append(i)
            j = i
        kept.reverse()
        costs.reverse()
        total = sum(costs)
        points = np.array([self._mm(chain[j]) for j in kept])
        points[0] = self.center
        if exact:
            points[-1] = np.asarray(point_mm, float)
        return total, points, costs


def move_field(
    center: Sequence[float],
    diameter: float,
    inches: int,
    rects=(),
    gaps=(),
    enemies: Sequence[tuple[Sequence[float], float]] = (),
    size_mm: Sequence[float] | None = None,
) -> MoveField:
    """Movement of a base of `diameter` at `center` with a budget of `inches`.

    `rects`: terrain rectangles (x0, y0, x1, y1), as `walls.obstacles`.
    `gaps`: rectangles of the open hatchways and breach points, as
    `walls.open_gaps`. `enemies`: (centre, diameter) of the bases that
    block. `size_mm`: board width and height, the base stays on the board."""
    k = MOVE_PX_PER_MM
    t = _tables(MM_PER_INCH * k)
    c, r = np.asarray(center, float), 0.5 * diameter
    clear = max(0.0, r - SLACK_MM)  # the centre keeps this distance to terrain
    reach = inches * MM_PER_INCH + r + 2.0
    lo, hi = np.floor(c - reach), np.ceil(c + reach)
    if size_mm is not None:
        lo, hi = np.maximum(lo, 0.0), np.minimum(hi, np.asarray(size_mm, float))
        lo, hi = np.minimum(lo, np.floor(c)), np.maximum(hi, np.ceil(c))  # a base that is off the board
    pad = t.half + 2
    origin = lo - pad / k
    w, h = (np.rint((hi - lo) * k).astype(int) + 1 + 2 * pad).tolist()

    def px(x: float, y: float) -> tuple[int, int]:
        return int(round((x - origin[0]) * k)), int(round((y - origin[1]) * k))

    free = np.zeros((h, w), np.uint8)
    if size_mm is None:
        free[:] = 1
    elif size_mm[0] > 2 * clear and size_mm[1] > 2 * clear:
        cv2.rectangle(free, px(clear, clear), px(size_mm[0] - clear, size_mm[1] - clear), 1, -1)
    grow = int(round(clear * k))
    for x0, y0, x1, y1 in np.asarray(rects, float).reshape(-1, 4):
        if x1 + clear < lo[0] or x0 - clear > hi[0] or y1 + clear < lo[1] or y0 - clear > hi[1]:
            continue
        cv2.rectangle(free, px(x0 - clear, y0), px(x1 + clear, y1), 0, -1)
        cv2.rectangle(free, px(x0, y0 - clear), px(x1, y1 + clear), 0, -1)
        for corner in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            cv2.circle(free, px(*corner), grow, 0, -1)
    barrier = np.zeros((h, w), np.uint8)
    zone = np.zeros((h, w), np.uint8)
    past = r + GAP_PAST_MM
    for x0, y0, x1, y1 in np.asarray(gaps, float).reshape(-1, 4):
        # The zone where the base can be over the wall while it goes
        # through the gap, and the line in the gap that costs 1 in to go over.
        if (x1 - x0) >= (y1 - y0):  # gap in a horizontal wall
            mid = 0.5 * (y0 + y1)
            ends = (px(x0, mid), px(x1, mid))
            cv2.rectangle(zone, px(x0, mid - past), px(x1, mid + past), 1, -1)
            cv2.rectangle(barrier, (ends[0][0], ends[0][1] - 1), ends[1], 1, -1)
        else:
            mid = 0.5 * (x0 + x1)
            ends = (px(mid, y0), px(mid, y1))
            cv2.rectangle(zone, px(mid - past, y0), px(mid + past, y1), 1, -1)
            cv2.rectangle(barrier, (ends[0][0] - 1, ends[0][1]), ends[1], 1, -1)
        for end in ends:
            cv2.circle(zone, end, int(round(past * k)), 1, -1)
    if len(np.asarray(gaps).reshape(-1)):
        for x0, y0, x1, y1 in np.asarray(rects, float).reshape(-1, 4):  # the centre itself stays out of the terrain
            cv2.rectangle(zone, px(x0 - GAP_WALL_MM, y0 - GAP_WALL_MM), px(x1 + GAP_WALL_MM, y1 + GAP_WALL_MM), 0, -1)
        if size_mm is not None:
            board = np.zeros((h, w), np.uint8)
            cv2.rectangle(board, px(0.0, 0.0), px(size_mm[0], size_mm[1]), 1, -1)
            zone &= board
    stand = free
    free = stand | zone
    for at, size in enemies:
        for mask in (stand, free):
            cv2.circle(mask, px(float(at[0]), float(at[1])), int(round((clear + 0.5 * size) * k)), 0, -1)
    for band in (stand, free, barrier):
        band[:pad], band[-pad:], band[:, :pad], band[:, -pad:] = 0, 0, 0, 0
    barrier &= free
    closed = free & ~barrier  # one step cannot go over a gap line
    doors = bool(barrier.any())
    wide = cv2.erode(closed, t.kernel, borderValue=0)  # a full step in each direction is free here
    near_door = cv2.dilate(barrier, t.kernel) if doors else None

    steps = np.full((h, w), UNREACHED, np.uint8)
    sx, sy = px(c[0], c[1])
    start = (sy, sx)
    steps[start] = 0
    escape = int(ESCAPE_MM * k)
    square = np.ones((3, 3), np.uint8)

    def stamp(mask: np.ndarray, ys: np.ndarray, xs: np.ndarray, level: int, out: int = 0) -> None:
        for i in range(0, len(ys), 256):
            ty, tx = _visible(mask, ys[i : i + 256], xs[i : i + 256], t, out)
            steps[ty, tx] = np.minimum(steps[ty, tx], level)

    # One straight increment from the start: the exact length, rounded up.
    # The steps below lose a part of a pixel at each step.
    seen, dist = _seen_from(closed, start, inches * MM_PER_INCH * k, escape)
    direct = np.ceil(np.maximum(dist[seen] / (MM_PER_INCH * k) - 1e-6, 0.0)).astype(np.uint8)
    steps[seen] = np.minimum(steps[seen], direct)
    steps[start] = 0

    for level in range(inches):
        if level == 0:
            ys, xs = np.array([sy]), np.array([sx])
            stamp(closed, ys, xs, 1, escape)
            if doors and inches >= 2 and near_door[start]:
                stamp(free, ys, xs, 2, escape)
            continue
        reached = steps <= level
        edge = reached & (cv2.dilate((free & ~reached).astype(np.uint8), square) > 0)
        if not edge.any():
            break
        # Far from terrain a step goes the same in all directions.
        far = cv2.dilate((edge & (wide > 0)).astype(np.uint8), t.kernel) > 0
        steps[far & (steps > level + 1)] = level + 1
        ys, xs = np.nonzero(edge & (wide == 0))
        stamp(closed, ys[::2], xs[::2], level + 1)
        if doors and level + 2 <= inches:
            at_door = near_door[ys, xs] > 0
            stamp(free, ys[at_door], xs[at_door], level + 2)
    steps[steps > inches] = UNREACHED
    return MoveField(steps, free, stand, barrier, origin, start, c, r, int(inches), k)
