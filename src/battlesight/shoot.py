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


@dataclass(frozen=True)
class Shot:
    """`origin`: start of the lines; `tangents` (2, 2): ends of the outer
    lines on the target rim; `pieces` (K, 4): rectangles x0, y0, x1, y1 of
    the intervening heavy terrain."""

    origin: np.ndarray
    tangents: np.ndarray
    pieces: np.ndarray
    obscured: bool
    in_cover: bool


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


def aim(
    origin: Sequence[float],
    target_center: Sequence[float],
    target_diameter: float,
    rects: np.ndarray,
    shooter: tuple[Sequence[float], float] | None = None,
    zone_mm: float = measure.MM_PER_INCH,
) -> Shot | None:
    """Targeting lines from `origin` to the target base and the heavy
    terrain `rects` (M, 4) in them. `shooter`: (centre, diameter) of the
    shooter's base, for its zone; None: the zone is around `origin`. None
    when `origin` is on the target base."""
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
    return Shot(o, t, np.asarray(pieces, float).reshape(-1, 4), obscured, in_cover)
