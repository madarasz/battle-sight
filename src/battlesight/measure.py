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
