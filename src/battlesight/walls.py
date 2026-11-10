"""Measurement around walls in board mm. No window, no camera.

On Tomb World a distance cannot be measured through walls, pillars, closed
hatchways and closed breach points. All these are axis-aligned rectangles
(`obstacles`). `Navigator` gives the shortest path between two base rims
that does not cross them (`path`), and the area that is within a path
length of a base rim (`reach`).

Distances are exact: a visibility graph on the rectangle corners with
Dijkstra. The measure line has no width, so it can touch a corner or run
along a wall face. Other bases do not block it.
"""

from __future__ import annotations

import heapq
from typing import Sequence

import cv2
import numpy as np

from . import measure

SOLID_KINDS = ("wall", "pillar", "wall_end")
OPENING_KINDS = ("hatchway", "breach")
RIM_SAMPLES = 64  # rim points of a base that a path can start or end at
REACH_RIM_SAMPLES = 16  # rim points that are sources of the range area (the centre is one more)
RANGE_PX_PER_MM = 4  # resolution of the range area mask
EPS_MM = 1e-6


def _box(points) -> tuple[float, float, float, float]:
    p = np.asarray(points, float).reshape(-1, 2)
    return float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())


EDGE_PAST_MM = 1.0  # an obstacle at the board edge goes this far past the edge


def obstacles(shapes, size_mm: Sequence[float] | None = None, outer: Sequence[float] | None = None) -> np.ndarray:
    """Rectangles (M, 4: x0, y0, x1, y1) that block a measurement, from
    `terrain.Shape` items. An open opening (`Shape.open`) cuts a gap of its
    width in its wall; a closed one blocks with its footprint.

    `size_mm` (board width, height) and `outer` (x0, y0, x1, y1 of the outer
    grid lines): terrain on an outer grid line blocks as far as the edge of
    the board. The printed margin between a pillar and the board edge is no
    way around it."""
    rects = [_box(s.points) for s in shapes if s.kind in SOLID_KINDS[1:] or (s.kind in OPENING_KINDS and not s.open)]
    gaps = [_box(s.points) for s in shapes if s.kind in OPENING_KINDS and s.open]
    for s in shapes:
        if s.kind != "wall":
            continue
        x0, y0, x1, y1 = _box(s.points)
        axis = 0 if (x1 - x0) >= (y1 - y0) else 1  # the wall runs along this axis
        lo, hi = (x0, x1) if axis == 0 else (y0, y1)
        pieces = [(lo, hi)]
        for g in gaps:
            cx, cy = 0.5 * (g[0] + g[2]), 0.5 * (g[1] + g[3])
            if not (x0 - EPS_MM <= cx <= x1 + EPS_MM and y0 - EPS_MM <= cy <= y1 + EPS_MM):
                continue
            g0, g1 = (g[0], g[2]) if axis == 0 else (g[1], g[3])
            pieces = [part for a, b in pieces for part in ((a, min(b, g0)), (max(a, g1), b))]
        for a, b in pieces:
            if b - a > EPS_MM:
                rects.append((a, y0, b, y1) if axis == 0 else (x0, a, x1, b))
    out = np.asarray(rects, float).reshape(-1, 4)
    if size_mm is not None and outer is not None and len(out):
        out[out[:, 0] <= outer[0] + EPS_MM, 0] = -EDGE_PAST_MM
        out[out[:, 1] <= outer[1] + EPS_MM, 1] = -EDGE_PAST_MM
        out[out[:, 2] >= outer[2] - EPS_MM, 2] = size_mm[0] + EDGE_PAST_MM
        out[out[:, 3] >= outer[3] - EPS_MM, 3] = size_mm[1] + EDGE_PAST_MM
    return out


def open_gaps(shapes) -> np.ndarray:
    """Rectangles (G, 4) of the open hatchways and breach points."""
    return np.asarray([_box(s.points) for s in shapes if s.kind in OPENING_KINDS and s.open], float).reshape(-1, 4)


class Navigator:
    """Shortest paths around a fixed set of rectangles. `size_mm` (width,
    height of the board) limits the range area, and a path does not bend
    at a corner that is off the board."""

    def __init__(self, rects, size_mm: Sequence[float] | None = None) -> None:
        self.rects = np.asarray(rects, float).reshape(-1, 4)
        self.size_mm = None if size_mm is None else (float(size_mm[0]), float(size_mm[1]))
        r = self.rects
        corners = np.concatenate([r[:, [0, 1]], r[:, [2, 1]], r[:, [2, 3]], r[:, [0, 3]]]) if len(r) else np.zeros((0, 2))
        corners = np.unique(np.round(corners, 6), axis=0) if len(corners) else corners
        keep = ~self._inside(corners)
        if self.size_mm is not None and len(corners):
            keep &= (corners >= -EPS_MM).all(axis=1) & (corners <= np.asarray(self.size_mm) + EPS_MM).all(axis=1)
        self.nodes = corners[keep]
        self._rows: dict[int, np.ndarray] = {}  # node -> distance to each node, inf when not visible

    # --- visibility ---------------------------------------------------------------

    def _inside(self, points: np.ndarray) -> np.ndarray:
        """True for each point in the open interior of a rectangle."""
        p, r = np.asarray(points, float).reshape(-1, 1, 2), self.rects[None]
        hit = (p[..., 0] > r[..., 0] + EPS_MM) & (p[..., 0] < r[..., 2] - EPS_MM) & (p[..., 1] > r[..., 1] + EPS_MM) & (p[..., 1] < r[..., 3] - EPS_MM)
        return hit.any(axis=1)

    def clear(self, p, q) -> np.ndarray:
        """For each point of `q` (K, 2): True when the segment from `p` does
        not go through the interior of a rectangle. A segment along a
        rectangle side or through a corner is clear."""
        p = np.asarray(p, float).reshape(2)
        q = np.asarray(q, float).reshape(-1, 2)
        if len(self.rects) == 0:
            return np.ones(len(q), bool)
        enter = np.zeros((len(q), len(self.rects)))
        leave = np.ones((len(q), len(self.rects)))
        with np.errstate(divide="ignore", invalid="ignore"):
            for axis in (0, 1):
                lo, hi = self.rects[:, axis] + EPS_MM, self.rects[:, axis + 2] - EPS_MM
                d = (q[:, axis] - p[axis])[:, None]
                t0, t1 = (lo - p[axis]) / d, (hi - p[axis]) / d
                within = (lo < p[axis]) & (p[axis] < hi)  # for a segment parallel to the slab
                parallel = np.abs(d) < 1e-12
                a = np.where(parallel, np.where(within, -np.inf, np.inf), np.minimum(t0, t1))
                b = np.where(parallel, np.where(within, np.inf, -np.inf), np.maximum(t0, t1))
                enter, leave = np.maximum(enter, a), np.minimum(leave, b)
        return ~(enter < leave - 1e-12).any(axis=1)

    def _row(self, i: int) -> np.ndarray:
        row = self._rows.get(i)
        if row is None:
            row = np.where(self.clear(self.nodes[i], self.nodes), np.linalg.norm(self.nodes - self.nodes[i], axis=1), np.inf)
            self._rows[i] = row
        return row

    def _spread(self, start: np.ndarray, limit: float = np.inf) -> tuple[np.ndarray, np.ndarray]:
        """Dijkstra on the corner graph from the start costs (inf = not a
        start). Gives the distance of each node and the node before it (-1 =
        first node of the path). Nodes beyond `limit` stay at inf."""
        dist = np.array(start, float)
        dist[dist > limit] = np.inf
        prev = np.full(len(dist), -1)
        done = np.zeros(len(dist), bool)
        heap = [(d, i) for i, d in enumerate(dist) if np.isfinite(d)]
        heapq.heapify(heap)
        while heap:
            d, i = heapq.heappop(heap)
            if done[i]:
                continue
            done[i] = True
            via = d + self._row(i)
            better = (via < dist) & (via <= limit) & ~done
            dist[better], prev[better] = via[better], i
            for j in np.flatnonzero(better):
                heapq.heappush(heap, (float(via[j]), int(j)))
        return dist, prev

    # --- bases --------------------------------------------------------------------

    def _rim(self, center, diameter: float, n: int = RIM_SAMPLES) -> np.ndarray:
        """Rim points of a base that its centre can see. This keeps out the
        part of a detected rim that is a little into a wall."""
        c = np.asarray(center, float)
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        rim = c + 0.5 * diameter * np.c_[np.cos(a), np.sin(a)]
        return rim[self.clear(c, rim)]

    def _links(self, points, center, diameter: float, rim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For each point: the length of the shortest clear segment to the
        base rim (inf when there is none) and its rim point. The candidates
        are the rim point on the line to the centre (exact) and `rim`."""
        points = np.asarray(points, float).reshape(-1, 2)
        c, r = np.asarray(center, float), 0.5 * diameter
        cost, ends = np.full(len(points), np.inf), points.copy()
        ok = len(rim) > 0  # no rim point: the centre is in a wall
        for k, p in enumerate(points):
            between = float(np.linalg.norm(p - c))
            if between <= r:
                cost[k] = 0.0 if ok else np.inf  # the point is on the base
                continue
            cand = np.concatenate([(c + r * (p - c) / between)[None], rim]) if ok else rim
            if len(cand) == 0:
                continue
            free = self.clear(p, cand)
            free[0] &= bool(self.clear(c, cand[:1])[0])  # the radial rim point must be a real one
            length = np.where(free, np.linalg.norm(cand - p, axis=1), np.inf)
            j = int(np.argmin(length))
            cost[k], ends[k] = length[j], cand[j]
        return cost, ends

    def _from_base(self, center, diameter: float, limit: float = np.inf):
        rim = self._rim(center, diameter)
        cost, ends = self._links(self.nodes, center, diameter, rim)
        dist, prev = self._spread(cost, limit)
        return rim, ends, dist, prev

    def path(self, center_a, diameter_a: float, center_b, diameter_b: float) -> tuple[float, np.ndarray] | None:
        """Shortest path between two base rims that crosses no rectangle:
        (length in mm, points (K, 2) from the rim of A to the rim of B).
        None when there is no path."""
        gap, start, end = measure.rim_gap(center_a, diameter_a, center_b, diameter_b)
        if self.clear(center_a, np.asarray(center_b, float)[None])[0]:
            return gap, np.array([start, end])
        rim_a, ends_a, dist, prev = self._from_base(center_a, diameter_a)
        rim_b = self._rim(center_b, diameter_b)
        best: tuple[float, np.ndarray] | None = None
        if len(rim_a):  # one segment between two rim points, not along the centre line
            cost, ends = self._links(rim_a, center_b, diameter_b, rim_b)
            j = int(np.argmin(cost))
            if np.isfinite(cost[j]):
                best = (float(cost[j]), np.array([rim_a[j], ends[j]]))
        if len(self.nodes):
            cost, ends = self._links(self.nodes, center_b, diameter_b, rim_b)
            total = dist + cost
            j = int(np.argmin(total))
            if np.isfinite(total[j]) and (best is None or total[j] < best[0]):
                points, i = [ends[j]], j
                while i >= 0:
                    points.append(self.nodes[i])
                    last, i = i, int(prev[i])
                points.append(ends_a[last])
                best = (float(total[j]), np.array(points[::-1]))
        return best

    def distance_to(self, center, diameter: float, point) -> float:
        """Path length from the base rim to one point (inf: no path)."""
        point = np.asarray(point, float).reshape(2)
        rim, _, dist, _ = self._from_base(center, diameter)
        best = float(self._links(point[None], center, diameter, rim)[0][0])
        if len(self.nodes):
            via = np.where(self.clear(point, self.nodes), dist + np.linalg.norm(self.nodes - point, axis=1), np.inf)
            best = min(best, float(via.min()))
        return best

    # --- range area ---------------------------------------------------------------

    def reach(self, center, diameter: float, range_mm: float) -> list[np.ndarray]:
        """Outline of the area with a path length of `range_mm` or less from
        the base rim: closed contours in board mm (N, 2). A contour inside
        another one is a hole (fill them together, even-odd)."""
        c, r, k = np.asarray(center, float), 0.5 * diameter, RANGE_PX_PER_MM
        lo, hi = c - (r + range_mm) - 1.0, c + (r + range_mm) + 1.0
        if self.size_mm is not None:
            lo, hi = np.maximum(lo, 0.0), np.minimum(hi, self.size_mm)
        if (hi - lo).min() <= 0:
            return []
        w, h = (np.ceil((hi - lo) * k).astype(int) + 1).tolist()
        mask = np.zeros((h, w), np.uint8)

        _, _, dist, _ = self._from_base(center, diameter, range_mm)
        sources = [(c, r + range_mm)] if not self._inside(c[None])[0] else []
        sources += [(p, range_mm) for p in self._rim(center, diameter, REACH_RIM_SAMPLES)]
        sources += [(self.nodes[i], range_mm - dist[i]) for i in np.flatnonzero(np.isfinite(dist))]
        for s, radius in sources:
            if radius * k < 1.0:
                continue
            # Work on the part of the mask that the disc of this source covers.
            x0, y0 = np.maximum(np.floor((s - radius - lo) * k).astype(int) - 1, 0)
            x1, y1 = np.minimum(np.ceil((s + radius - lo) * k).astype(int) + 2, (w, h))
            if x1 <= x0 or y1 <= y0:
                continue
            origin = lo + np.array([x0, y0]) / k

            def px(points) -> np.ndarray:
                return np.rint((np.asarray(points, float) - origin) * k).astype(np.int32)

            part = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.circle(part, tuple(px(s).tolist()), int(round(radius * k)), 255, -1)
            far = 1.2 * radius + 10.0
            near = (self.rects[:, 0] < s[0] + radius) & (self.rects[:, 2] > s[0] - radius) & (self.rects[:, 1] < s[1] + radius) & (self.rects[:, 3] > s[1] - radius)
            for rx0, ry0, rx1, ry1 in self.rects[near]:
                box = np.array([[rx0, ry0], [rx1, ry0], [rx1, ry1], [rx0, ry1]])
                cv2.fillPoly(part, [px(box)], 0)
                for a, b in zip(box, np.roll(box, -1, axis=0)):
                    da, db = a - s, b - s
                    if abs(da[0] * db[1] - da[1] * db[0]) < 1e-6:
                        continue  # the source is on the line of this side: no shadow
                    # The area behind this side: between the two rays, out to an arc
                    # beyond the disc. A straight far side cuts a wide shadow short.
                    start = np.arctan2(db[1], db[0])
                    turn = (np.arctan2(da[1], da[0]) - start + np.pi) % (2 * np.pi) - np.pi
                    angles = start + turn * np.linspace(0.0, 1.0, int(np.ceil(abs(turn) / (np.pi / 8))) + 1)
                    shadow = np.concatenate([[a, b], s + far * np.c_[np.cos(angles), np.sin(angles)]])
                    cv2.fillPoly(part, [px(shadow)], 0)
            np.maximum(mask[y0:y1, x0:x1], part, out=mask[y0:y1, x0:x1])
        cv2.circle(mask, tuple(np.rint((c - lo) * k).astype(int).tolist()), int(round(r * k)), 255, -1)  # the base

        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        return [lo + cnt.reshape(-1, 2) / k for cnt in contours if len(cnt) >= 3]


def contains(contours: list[np.ndarray], point) -> bool:
    """True when the point is in the area of `Navigator.reach` (even-odd)."""
    p = (float(point[0]), float(point[1]))
    return sum(cv2.pointPolygonTest(c.astype(np.float32).reshape(-1, 1, 2), p, False) >= 0 for c in contours) % 2 == 1
