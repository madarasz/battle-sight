"""Targeting lines and intervening terrain in board mm. No window, no camera.

The shooter draws straight lines, 1 mm in diameter, from one point of its
base to each facing part of the target's base. The point is the mouse
pointer. The lines fill the area between the pointer and the two points
where the lines touch the target rim (`tangent_points`): a triangle, less
the part of the target base in it.

Heavy terrain is walls, pillars and closed hatchways and breach points
(`walls.obstacles`: an open opening cuts a gap in its wall). A piece of
heavy terrain that the lines touch is intervening. Its parts in the lines
decide the result (`aim`):

- a part outside the 1 in zones around both the target and the shooter: obscured;
- a part in the 1 in zone around the target: in cover.

When the pointer is not on a base, the shooter zone is 1 in around the pointer.

Light terrain (polygons, `terrain.light_outline`) gives cover in the same
way, and never obscures. No terrain gives cover while the target is within
2 in of the shooter (rim to rim; from the pointer without a shooter base).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import measure

LINE_RADIUS_MM = 0.5  # the targeting lines are 1 mm in diameter
SAMPLE_MM = 0.5  # test points in a terrain piece are this far apart
ARC_POINTS = 24  # points of the facing rim in `facing_outline`
NO_COVER_MM = 2 * measure.MM_PER_INCH  # no cover while the target is nearer to the shooter than this


@dataclass(frozen=True)
class Shot:
    """`origin`: start of the lines; `tangents` (2, 2): ends of the outer
    lines on the target rim; `pieces` (K, 4): rectangles x0, y0, x1, y1 of
    the intervening heavy terrain; `light`: indexes of the light terrain
    polygons that give cover; `cover_by`: "heavy", "light", "both" or None;
    `too_near`: terrain is in the target zone, but the target is within
    2 in of the shooter, so it is not in cover."""

    origin: np.ndarray
    tangents: np.ndarray
    pieces: np.ndarray
    obscured: bool
    in_cover: bool
    light: tuple[int, ...] = ()
    cover_by: str | None = None
    too_near: bool = False


def tangent_points(origin: Sequence[float], center: Sequence[float], diameter: float) -> np.ndarray | None:
    """The two points (2, 2) where lines from `origin` touch the base rim.
    None when `origin` is on the base."""
    o, c, r = np.asarray(origin, float), np.asarray(center, float), 0.5 * diameter
    d = float(np.linalg.norm(o - c))
    if d <= r:
        return None
    u = (o - c) / d
    alpha = math.acos(r / d)
    out = []
    for a in (alpha, -alpha):
        ca, sa = math.cos(a), math.sin(a)
        out.append(c + r * np.array([ca * u[0] - sa * u[1], sa * u[0] + ca * u[1]]))
    return np.array(out)


def facing_outline(origin: Sequence[float], center: Sequence[float], diameter: float, n: int = ARC_POINTS) -> np.ndarray | None:
    """Outline (N, 2) of the area of the targeting lines: the origin, then
    the facing rim of the target from one tangent point to the other."""
    o, c, r = np.asarray(origin, float), np.asarray(center, float), 0.5 * diameter
    d = float(np.linalg.norm(o - c))
    if d <= r:
        return None
    toward, alpha = math.atan2(o[1] - c[1], o[0] - c[0]), math.acos(r / d)
    a = np.linspace(toward + alpha, toward - alpha, n)  # same order as `tangent_points`
    return np.vstack([np.asarray(origin, float), c + r * np.c_[np.cos(a), np.sin(a)]])


def _in_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    def side(u, v):
        return (v[0] - u[0]) * (p[:, 1] - u[1]) - (v[1] - u[1]) * (p[:, 0] - u[0])

    s1, s2, s3 = side(a, b), side(b, c), side(c, a)
    return ((s1 >= 0) & (s2 >= 0) & (s3 >= 0)) | ((s1 <= 0) & (s2 <= 0) & (s3 <= 0))


def _touches_polygon(p: np.ndarray, poly: np.ndarray, reach: float) -> np.ndarray:
    """Which points (N, 2) are in the polygon (M, 2) or within `reach` of its edge."""
    a, b = poly, np.roll(poly, -1, axis=0)
    px, py = p[:, 0][:, None], p[:, 1][:, None]
    ax, ay, bx, by = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        crosses = ((ay > py) != (by > py)) & (px < ax + (py - ay) * (bx - ax) / (by - ay))
    inside = crosses.sum(axis=1) % 2 == 1  # even-odd rule
    d, rel = b - a, p[:, None, :] - a[None, :, :]
    t = np.clip((rel * d).sum(axis=2) / np.maximum((d * d).sum(axis=1), 1e-12), 0.0, 1.0)
    near = np.linalg.norm(rel - t[:, :, None] * d[None, :, :], axis=2).min(axis=1) <= reach
    return inside | near


def aim(
    origin: Sequence[float],
    target_center: Sequence[float],
    target_diameter: float,
    rects: np.ndarray,
    shooter: tuple[Sequence[float], float] | None = None,
    zone_mm: float = measure.MM_PER_INCH,
    light: Sequence[np.ndarray] = (),
) -> Shot | None:
    """Targeting lines from `origin` to the target base and the heavy
    terrain `rects` (M, 4) in them. `shooter`: (centre, diameter) of the
    shooter's base, for its zone; None: the zone is around `origin`.
    `light`: outlines (N, 2) of the light terrain. None when `origin` is on
    the target base."""
    o = np.asarray(origin, float)
    tc, tr = np.asarray(target_center, float), 0.5 * target_diameter
    t = tangent_points(o, tc, target_diameter)
    if t is None:
        return None
    lo, hi = np.minimum(o, t.min(axis=0)), np.maximum(o, t.max(axis=0))
    pieces, obscured, in_cover = [], False, False
    for rect in np.asarray(rects, float).reshape(-1, 4):
        x0, y0, x1, y1 = rect + (-LINE_RADIUS_MM, -LINE_RADIUS_MM, LINE_RADIUS_MM, LINE_RADIUS_MM)
        if x1 < lo[0] or x0 > hi[0] or y1 < lo[1] or y0 > hi[1]:
            continue
        xs = np.linspace(x0, x1, max(2, math.ceil((x1 - x0) / SAMPLE_MM) + 1))
        ys = np.linspace(y0, y1, max(2, math.ceil((y1 - y0) / SAMPLE_MM) + 1))
        p = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
        to_target = np.linalg.norm(p - tc, axis=1)
        inside = _in_triangle(p, o, t[0], t[1]) & (to_target > tr)
        if not inside.any():
            continue
        pieces.append(rect)
        near_target = to_target <= tr + zone_mm
        s_center, s_diameter = (o, 0.0) if shooter is None else (np.asarray(shooter[0], float), shooter[1])
        near_shooter = np.linalg.norm(p - s_center, axis=1) <= 0.5 * s_diameter + zone_mm
        in_cover |= bool((inside & near_target).any())
        obscured |= bool((inside & ~near_target & ~near_shooter).any())
    covering = []
    for k, poly in enumerate(light):
        poly = np.asarray(poly, float).reshape(-1, 2)
        x0, y0 = poly.min(axis=0) - LINE_RADIUS_MM
        x1, y1 = poly.max(axis=0) + LINE_RADIUS_MM
        if x1 < lo[0] or x0 > hi[0] or y1 < lo[1] or y0 > hi[1]:
            continue
        # Only the part of the box in the target zone can give cover.
        x0, y0 = max(x0, tc[0] - tr - zone_mm), max(y0, tc[1] - tr - zone_mm)
        x1, y1 = min(x1, tc[0] + tr + zone_mm), min(y1, tc[1] + tr + zone_mm)
        if x1 < x0 or y1 < y0:
            continue
        xs = np.linspace(x0, x1, max(2, math.ceil((x1 - x0) / SAMPLE_MM) + 1))
        ys = np.linspace(y0, y1, max(2, math.ceil((y1 - y0) / SAMPLE_MM) + 1))
        p = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
        to_target = np.linalg.norm(p - tc, axis=1)
        p = p[_in_triangle(p, o, t[0], t[1]) & (to_target > tr) & (to_target <= tr + zone_mm)]
        if len(p) and _touches_polygon(p, poly, LINE_RADIUS_MM).any():
            covering.append(k)
    if shooter is None:
        gap = float(np.linalg.norm(o - tc)) - tr
    else:
        gap = float(np.linalg.norm(np.asarray(shooter[0], float) - tc)) - tr - 0.5 * shooter[1]
    by = "both" if in_cover and covering else "heavy" if in_cover else "light" if covering else None
    too_near = by is not None and gap < NO_COVER_MM
    if too_near:
        by, covering = None, []
    return Shot(o, t, np.asarray(pieces, float).reshape(-1, 4), obscured, by is not None, tuple(covering), by, too_near)
