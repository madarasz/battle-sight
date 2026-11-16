"""Position and rotation of the light terrain pieces in a top-down image.

A piece file holds the outline of the footprint (skill `record-terrain`).
The map file gives the footprint of the piece on the layout card: the real
footprint must be in that rectangle plus a margin. `locate` searches the
positions and angles that this permits for the outline, with the edges of
the image only. It does not use the colour of the piece: each owner paints
the pieces differently.

Score of a pose: the part of the outline that lies on an image edge with
the direction of the outline (`edge_maps`: Canny edges of the Lab channels,
in `BINS` direction classes, each grown by `TOL_MM`). A piece with much
detail has edges everywhere, so the result is this part against the usual
level at that place: (best - median of the best of each angle) /
(1 - median). A coarse pass over all angles, then a fine pass around the
best result.

No window, no camera. Measured on two test photos with the C2 outline
(2026-11): see `specs/measure-and-move.md` section 10. An image of the
piece as a template gave similar results but needs a photo of each owner's
paint; a colour mask of "not board" pixels was not usable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

PX_PER_MM = 2.0  # scale of the edge maps, at most (not above the image)
BINS = 12  # direction classes of an edge, over 180 degrees
TOL_MM = 0.75  # an outline point is on an edge within about this
CANNY = (60, 150)  # thresholds on each Lab channel after `BLUR_PX`
BLUR_PX = 1.0
COARSE_STEP_DEG, COARSE_STEP_MM, COARSE_POINT_MM = 4.0, 1.0, 1.0
FINE_STEP_DEG, FINE_SPAN_DEG = 0.5, 4.0  # fine pass: this far each side of the coarse angle
FINE_STEP_MM, FINE_REACH_MM, FINE_POINT_MM = 0.5, 2.0, 0.5
MIN_ANGLES = 8  # fewer possible angles: no usual level, the piece is not confirmed


@dataclass(frozen=True)
class Pose:
    center_mm: tuple[float, float]  # origin of the piece frame on the board (footprint)
    angle_deg: float
    score: float  # outline on edges, against the usual level at that place; 1 = all of it
    confirmed: bool  # False: the piece was not found; centre and angle are the nominal pose


def edge_maps(image: np.ndarray, px_per_mm: float) -> np.ndarray:
    """(BINS, h, w) uint8: 1 where an edge of that direction class (or of a
    class next to it) is within `TOL_MM`."""
    lab = cv2.cvtColor(cv2.GaussianBlur(image, (0, 0), BLUR_PX), cv2.COLOR_BGR2LAB)
    edges = np.zeros(lab.shape[:2], bool)
    best = np.zeros(lab.shape[:2], np.float32)
    direction = np.zeros(lab.shape[:2], np.float32)
    for c in range(3):
        ch = np.ascontiguousarray(lab[:, :, c])
        edges |= cv2.Canny(ch, *CANNY) > 0
        gx, gy = cv2.Sobel(ch, cv2.CV_32F, 1, 0), cv2.Sobel(ch, cv2.CV_32F, 0, 1)
        mag = gx * gx + gy * gy
        direction = np.where(mag > best, np.arctan2(gy, gx), direction)
        best = np.maximum(mag, best)
    cls = np.floor((direction % math.pi) / math.pi * BINS).astype(int) % BINS
    k = 2 * max(1, round(TOL_MM * px_per_mm)) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    maps = np.zeros((BINS, *edges.shape), np.uint8)
    for b in range(BINS):
        near = edges & ((cls == b) | (cls == (b + 1) % BINS) | (cls == (b - 1) % BINS))
        maps[b] = cv2.dilate(near.astype(np.uint8), kernel)
    return maps


def outline_points(outline: np.ndarray, angle_deg: float, step_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Points (P, 2) along the outline turned by `angle_deg` (as
    `terrain.Piece.at`), about `step_mm` apart, and the direction class of
    the outline normal at each."""
    a = math.radians(angle_deg)
    p = np.asarray(outline, float) @ np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]).T
    pts, cls = [], []
    for u, v in zip(p, np.roll(p, -1, axis=0)):
        d = v - u
        length = float(np.linalg.norm(d))
        if length < 1e-9:
            continue
        n = max(1, round(length / step_mm))
        pts.append(u + ((np.arange(n) + 0.5) / n)[:, None] * d)
        normal = math.atan2(d[0], -d[1]) % math.pi
        cls.append(np.full(n, int(normal / math.pi * BINS) % BINS))
    return np.vstack(pts), np.concatenate(cls)


def origin_box(outline: np.ndarray, angle_deg: float, rect: Sequence[float]) -> tuple[float, float, float, float] | None:
    """Where the origin of a piece can be at `angle_deg`, so that its
    outline (N, 2, piece frame) stays in `rect` (x0, y0, x1, y1). None: it
    does not fit at this angle."""
    a = math.radians(angle_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    pts = np.asarray(outline, float) @ rot.T
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    x0, y0, x1, y1 = rect[0] - lo[0], rect[1] - lo[1], rect[2] - hi[0], rect[3] - hi[1]
    return None if x1 < x0 or y1 < y0 else (float(x0), float(y0), float(x1), float(y1))


def _best(maps: np.ndarray, px_per_mm: float, lo_mm: np.ndarray, outline: np.ndarray, angle: float, box, step_mm: float, point_mm: float) -> tuple[float, tuple[float, float]]:
    """Best (score, origin in board mm) at one angle, the origin on a grid
    of `step_mm` in `box` (x0, y0, x1, y1 in board mm). `maps` pixel (0, 0)
    is at `lo_mm`. An outline point outside the maps is not on an edge."""
    xs = np.arange(box[0], box[2] + 1e-6, step_mm)
    ys = np.arange(box[1], box[3] + 1e-6, step_mm)
    pts, cls = outline_points(outline, angle, point_mm)
    ix = np.rint((pts[:, 0][:, None] + xs[None, :] - lo_mm[0]) * px_per_mm).astype(int)  # (P, X)
    iy = np.rint((pts[:, 1][:, None] + ys[None, :] - lo_mm[1]) * px_per_mm).astype(int)  # (P, Y)
    h, w = maps.shape[1:]
    inside = ((iy >= 0) & (iy < h))[:, :, None] & ((ix >= 0) & (ix < w))[:, None, :]
    hits = maps[cls[:, None, None], np.clip(iy, 0, h - 1)[:, :, None], np.clip(ix, 0, w - 1)[:, None, :]] * inside
    score = hits.mean(axis=0)  # (Y, X)
    j, i = np.unravel_index(int(np.argmax(score)), score.shape)
    return float(score[j, i]), (float(xs[i]), float(ys[j]))


def locate(
    image: np.ndarray,
    px_per_mm: float,
    origin_mm: Sequence[float],
    outline: np.ndarray,
    rect: Sequence[float],
    nominal_mm: Sequence[float],
    nominal_deg: float,
    min_score: float = 0.45,
) -> Pose:
    """Find one piece in the top-down `image` (`px_per_mm`, pixel (0, 0) at
    board mm `origin_mm`): the pose of `outline` (piece frame) in `rect`
    (x0, y0, x1, y1 in board mm: the footprint of the layout card plus a
    margin) that lies best on image edges. Not confirmed (the nominal pose)
    when the score is under `min_score`."""
    nominal = (float(nominal_mm[0]), float(nominal_mm[1]))
    outline = np.asarray(outline, float)
    scale = min(PX_PER_MM, px_per_mm)
    # The part of the image around the rectangle, at the scale of the edge maps.
    lo = np.array([rect[0], rect[1]], float) - 2 * TOL_MM
    size = np.ceil((np.array([rect[2], rect[3]], float) + 2 * TOL_MM - lo) * scale).astype(int) + 1
    s = px_per_mm / scale
    off = (lo - np.asarray(origin_mm, float)) * px_per_mm
    M = np.array([[s, 0, off[0]], [0, s, off[1]]])  # window px -> image px
    window = cv2.warpAffine(image, M, (int(size[0]), int(size[1])), flags=(cv2.INTER_AREA if s > 1 else cv2.INTER_LINEAR) | cv2.WARP_INVERSE_MAP)
    maps = edge_maps(window, scale)

    coarse = []  # (score, angle, origin)
    for angle in np.arange(0.0, 360.0, COARSE_STEP_DEG):
        box = origin_box(outline, angle, rect)
        if box is not None:
            score, at = _best(maps, scale, lo, outline, angle, box, COARSE_STEP_MM, COARSE_POINT_MM)
            coarse.append((score, float(angle), at))
    if len(coarse) < MIN_ANGLES:
        return Pose(nominal, float(nominal_deg), 0.0, False)
    usual = float(np.median([c[0] for c in coarse]))
    raw, angle, at = max(coarse)
    for fine in angle + np.arange(-FINE_SPAN_DEG, FINE_SPAN_DEG + 1e-6, FINE_STEP_DEG):
        box = origin_box(outline, fine, rect)
        if box is None:
            continue
        near = (max(box[0], at[0] - FINE_REACH_MM), max(box[1], at[1] - FINE_REACH_MM), min(box[2], at[0] + FINE_REACH_MM), min(box[3], at[1] + FINE_REACH_MM))
        if near[2] < near[0] or near[3] < near[1]:
            continue
        score, fine_at = _best(maps, scale, lo, outline, fine, near, FINE_STEP_MM, FINE_POINT_MM)
        if score > raw:
            raw, angle, at = score, float(fine), fine_at
    score = (raw - usual) / max(1.0 - usual, 1e-6)
    if score < min_score:
        return Pose(nominal, float(nominal_deg), score, False)
    return Pose(at, float(angle % 360.0), score, True)
