"""Rings and rim-to-rim distances in board mm (plan priority 5). No window, no camera.

All measurement is 2D in the board plane and goes to the base rim, never to
the model.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

MM_PER_INCH = 25.4


def base_at(centers_mm: Sequence, diameters_mm: Sequence[float], at_mm: Sequence[float], reach_mm: float = 5.0) -> int | None:
    """Index of the base whose disc (plus `reach_mm`) holds the point, nearest centre first."""
    if len(centers_mm) == 0:
        return None
    d = np.linalg.norm(np.asarray(centers_mm, float).reshape(-1, 2) - np.asarray(at_mm, float), axis=1)
    i = int(np.argmin(d))
    return i if d[i] <= 0.5 * diameters_mm[i] + reach_mm else None


def path_distance(points_mm, at_mm) -> float:
    """Shortest distance from a chain of straight segments to a point."""
    pts, at = np.asarray(points_mm, float).reshape(-1, 2), np.asarray(at_mm, float)
    best = float(np.linalg.norm(pts[0] - at))
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        t = np.clip(np.dot(at - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(a + t * ab - at)))
    return best


def ring_radius_mm(diameter_mm: float, inches: float) -> float:
    """Radius of the ring that is `inches` from the rim of a base."""
    return 0.5 * diameter_mm + inches * MM_PER_INCH


def rim_gap(center_a: Sequence[float], diameter_a: float, center_b: Sequence[float], diameter_b: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Shortest distance between two base rims, and its end points on the rims.

    Touching or overlapping bases have a distance of 0, and both end points
    are the same point."""
    a, b = np.asarray(center_a, float), np.asarray(center_b, float)
    between = float(np.linalg.norm(b - a))
    unit = (b - a) / between if between > 0.0 else np.array([1.0, 0.0])
    gap = max(0.0, between - 0.5 * diameter_a - 0.5 * diameter_b)
    start = a + 0.5 * diameter_a * unit
    return gap, start, start + gap * unit


# --- control marks ----------------------------------------------------------------


def melee_pairs(centers_mm: Sequence, diameters_mm: Sequence[float], teams: Sequence[str | None], reach_mm: float = MM_PER_INCH) -> list[tuple[int, int]]:
    """Pairs (i, j), i < j, of bases of two different teams with rims at most
    `reach_mm` apart: the operatives are in melee. A base with team None
    (no name) is in no pair."""
    out = []
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            if teams[i] is None or teams[j] is None or teams[i] == teams[j]:
                continue
            if rim_gap(centers_mm[i], diameters_mm[i], centers_mm[j], diameters_mm[j])[0] <= reach_mm:
                out.append((i, j))
    return out


def objective_contacts(centers_mm: Sequence, diameters_mm: Sequence[float], objectives_mm: Sequence, objective_mm: float, reach_mm: float = MM_PER_INCH) -> list[tuple[int, int]]:
    """Pairs (base index, objective index) where the base rim is at most
    `reach_mm` from the edge of the objective marker (diameter `objective_mm`)."""
    return [
        (i, k)
        for i in range(len(diameters_mm))
        for k in range(len(objectives_mm))
        if rim_gap(centers_mm[i], diameters_mm[i], objectives_mm[k], objective_mm)[0] <= reach_mm
    ]
