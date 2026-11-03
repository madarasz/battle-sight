"""Round bases (and later tokens) of known diameters in the top-down image.

Priority 3 of the plan. Pure geometry, no training:

1. Optional foreground mask: difference to an empty-board reference image,
   rectified the same way. Without a reference the whole board is searched.
2. Proposals: every strong edge pixel votes for the centre that lies one
   radius inward along its gradient (dark rim inside, lighter board outside),
   once for each known diameter. Peaks of the vote maps are the proposals.
3. Verification: rays from the proposed centre. On each ray the rim edge is
   the strongest dark-to-light step near the expected radius, with a dark
   band inside it. A robust circle fit to these edge points gives the centre
   and diameter. `coverage` is the part of the rim that supports the fit, so
   a model or a wall can hide part of the rim.
4. The fitted diameter picks the nearest size class. The centre is fitted
   again with the radius of that class.

The diameter is the base bottom, where it touches the board. The bottom edge
lies on the play surface, so the board plane homography maps it without
parallax. The top of the base is smaller and is not used.

All functions work on one image in rectified pixels: px = (mm + origin_mm) * px_per_mm.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import BoardConfig

N_RAYS = 120
# Rim band samples inside the edge. The visible black bevel is 1 to 3 mm wide:
# narrow on the side away from the camera, so the samples stay near the edge.
RIM_IN_MM = (0.5, 1.0)
RIM_OUT_MM = 1.25  # board sample outside the edge


@dataclass(frozen=True)
class CircleParams:
    diameters_mm: tuple[float, ...]
    px_per_mm: float
    origin_mm: float = 0.0  # board origin in the image, in mm from the image corner
    diameter_tol_mm: float = 3.0  # fitted diameter must be this close to a class
    min_coverage: float = 0.5  # part of the rim with edge support
    min_contrast: float = 12.0  # L (0..255) step from rim to board, at least
    rim_ratio: float = 0.75  # rim band L at most this part of the board L just outside
    search_mm: float = 3.5  # ray search band around the proposal radius
    max_proposals: int = 150  # per diameter
    min_foreground: float = 0.25  # with a reference: part of the disc that changed

    @classmethod
    def from_config(cls, cfg: BoardConfig, origin_mm: float) -> CircleParams:
        return cls(
            diameters_mm=cfg.bases_diameters_mm,
            px_per_mm=cfg.bases_px_per_mm,
            origin_mm=origin_mm,
            diameter_tol_mm=cfg.bases_diameter_tol_mm,
            min_coverage=cfg.bases_min_coverage,
            min_contrast=cfg.bases_min_contrast,
            rim_ratio=cfg.bases_rim_ratio,
            min_foreground=cfg.bases_min_foreground,
        )


@dataclass
class Circle:
    center_mm: np.ndarray  # (2,) board mm
    diameter_mm: float  # size class
    fit_diameter_mm: float  # free fit, before the size class was chosen
    coverage: float  # 0..1, part of the rim that supports the fit
    rms_mm: float  # residual of the supporting edge points
    contrast: float  # median L step from the rim band to the board outside

    def to_dict(self) -> dict:
        return {
            "center_mm": [round(float(v), 2) for v in self.center_mm],
            "diameter_mm": self.diameter_mm,
            "fit_diameter_mm": round(self.fit_diameter_mm, 2),
            "coverage": round(self.coverage, 3),
            "rms_mm": round(self.rms_mm, 3),
            "contrast": round(self.contrast, 1),
        }


def lightness(img: np.ndarray) -> np.ndarray:
    """L channel of Lab, 0..255, float32."""
    if img.ndim == 2:
        return img.astype(np.float32)
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)


def foreground_mask(img: np.ndarray, reference: np.ndarray, threshold: float = 25.0) -> np.ndarray:
    """Pixels that differ from the empty-board reference (bool, same size).
    Distance in Lab after a small blur, so noise and small shifts do not count."""
    a = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32), (0, 0), 2.0)
    b = cv2.GaussianBlur(cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32), (0, 0), 2.0)
    d = np.linalg.norm(a - b, axis=2)
    fg = (d > threshold).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return fg > 0


def _gradients(L: np.ndarray, px_per_mm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = cv2.GaussianBlur(L, (0, 0), max(0.8, 0.35 * px_per_mm))
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return g, gx, gy


def _rim_test(inner: np.ndarray, inner2: np.ndarray, outer: np.ndarray, p: CircleParams) -> np.ndarray:
    """Dark rim band inside the edge, lighter board outside. `inner` and
    `inner2` are L at two depths into the rim band: a thin printed line is
    dark at one of them only."""
    rim = np.maximum(inner, inner2)
    return (outer - rim > p.min_contrast) & (rim < p.rim_ratio * outer)


def _proposals(g: np.ndarray, gx: np.ndarray, gy: np.ndarray, p: CircleParams, allowed: np.ndarray, edge_min: float) -> list[tuple[float, float, float]]:
    """(x, y, diameter) proposals from gradient votes, best first per diameter.
    Only edge pixels that pass the rim test vote."""
    mag = np.hypot(gx, gy)
    ys, xs = np.nonzero(mag > edge_min)
    nx, ny = gx[ys, xs] / mag[ys, xs], gy[ys, xs] / mag[ys, xs]
    s = p.px_per_mm

    def at(depth_mm: float) -> np.ndarray:  # nearest pixel is enough for a vote
        h, w = g.shape
        x = np.clip(np.rint(xs + depth_mm * s * nx).astype(np.int64), 0, w - 1)
        y = np.clip(np.rint(ys + depth_mm * s * ny).astype(np.int64), 0, h - 1)
        return g[y, x]

    ok = _rim_test(at(-RIM_IN_MM[0]), at(-RIM_IN_MM[1]), at(RIM_OUT_MM), p)
    xs, ys, nx, ny = xs[ok], ys[ok], nx[ok], ny[ok]
    small = allowed[::2, ::2]  # votes go to a half-resolution map
    out = []
    for d in p.diameters_mm:
        acc = np.zeros(small.shape, np.float32)
        for r_mm in (0.5 * d - 0.75, 0.5 * d, 0.5 * d + 0.75):
            r = r_mm * p.px_per_mm
            cx = np.rint((xs - r * nx) / 2).astype(np.int64)
            cy = np.rint((ys - r * ny) / 2).astype(np.int64)
            ok = (cx >= 0) & (cx < acc.shape[1]) & (cy >= 0) & (cy < acc.shape[0])
            np.add.at(acc, (cy[ok], cx[ok]), 1.0)
        acc = cv2.GaussianBlur(acc, (0, 0), 1.0)
        acc[~small] = 0
        k = int(0.25 * d * p.px_per_mm) | 1
        peaks = np.argwhere((acc == cv2.dilate(acc, np.ones((k, k), np.uint8))) & (acc > 0))
        vals = acc[peaks[:, 0], peaks[:, 1]]
        for i in np.argsort(-vals)[: p.max_proposals]:
            y, x = peaks[i] * 2
            out.append((float(x), float(y), float(d)))
    return out


def _sample(img: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return cv2.remap(img, X.astype(np.float32), Y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _ray_edges(g: np.ndarray, x0: float, y0: float, r: float, p: CircleParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rim edge on each ray from (x0, y0) near radius r.

    Returns (radius px per ray, NaN where no rim edge; rim band L; board L outside)."""
    s = p.px_per_mm
    a = np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False)
    ca, sa = np.cos(a)[:, None], np.sin(a)[:, None]
    step = 0.5
    k_in = [int(round(v * s / step)) for v in RIM_IN_MM]
    k_out = int(round(RIM_OUT_MM * s / step))
    lo = max(r - p.search_mm * s, 2.0) - k_in[1] * step
    rs = np.arange(lo, r + p.search_mm * s + (k_out + 1) * step, step)
    prof = _sample(g, x0 + ca * rs[None, :], y0 + sa * rs[None, :])
    d = np.gradient(prof, axis=1) / step  # L per px, outward
    inner = np.maximum(np.roll(prof, k_in[0], axis=1), np.roll(prof, k_in[1], axis=1))  # rim band, inside each sample
    outer = np.roll(prof, -k_out, axis=1)
    ok = _rim_test(inner, inner, outer, p) & (d > 0)
    ok[:, : k_in[1]] = ok[:, -k_out:] = False
    score = np.where(ok, d, 0.0)
    j = score.argmax(axis=1)
    rows = np.arange(N_RAYS)
    found = score[rows, j] > 0
    jj = np.clip(j, 1, len(rs) - 2)
    a_, b_, c_ = d[rows, jj - 1], d[rows, jj], d[rows, jj + 1]
    den = a_ - 2 * b_ + c_
    with np.errstate(divide="ignore", invalid="ignore"):
        off = np.where(np.abs(den) > 1e-9, 0.5 * (a_ - c_) / den, 0.0)
    rad = rs[jj] + 0.5 * np.clip(off, -1, 1)
    rad = np.where(found, rad, np.nan)
    return rad, inner[rows, j], outer[rows, j]


def _fit_circle(pts: np.ndarray, r_fixed: float | None = None, center: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Least squares circle through pts (N, 2). With r_fixed only the centre moves."""
    if r_fixed is None:
        A = np.c_[2 * pts, np.ones(len(pts))]
        b = (pts**2).sum(axis=1)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        c = sol[:2]
        return c, float(np.sqrt(max(sol[2] + c @ c, 1e-9)))
    c = pts.mean(axis=0) if center is None else center.astype(np.float64).copy()
    for _ in range(10):  # Gauss-Newton on |p - c| - r
        v = pts - c
        n = np.linalg.norm(v, axis=1)
        n = np.maximum(n, 1e-9)
        res = n - r_fixed
        J = -v / n[:, None]
        step, *_ = np.linalg.lstsq(J, -res, rcond=None)
        c = c + step
        if np.linalg.norm(step) < 1e-3:
            break
    return c, r_fixed


def _verify(g: np.ndarray, x0: float, y0: float, d_mm: float, p: CircleParams) -> Circle | None:
    s = p.px_per_mm
    a = np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False)
    dirs = np.c_[np.cos(a), np.sin(a)]
    c = np.array([x0, y0], dtype=np.float64)
    r = 0.5 * d_mm * s
    tol = 1.0 * s  # edge point within 1 mm of the circle supports it
    free_r = r
    for it in range(3):  # rays from the better centre each round
        rad, inner, outer = _ray_edges(g, c[0], c[1], r if it == 0 else free_r, p)
        good = ~np.isnan(rad)
        if good.sum() < 0.3 * N_RAYS:
            return None
        pts = c + dirs[good] * rad[good, None]
        keep = np.ones(len(pts), bool)
        cc, rr = c, free_r
        for _ in range(4):  # drop outliers: model and wall edges
            if keep.sum() < 0.3 * N_RAYS:
                return None
            cc, rr = _fit_circle(pts[keep])
            res = np.abs(np.linalg.norm(pts - cc, axis=1) - rr)
            keep = res < max(tol, 2.5 * np.median(res[keep]) * 1.4826)
        if np.linalg.norm(cc - c) > 0.5 * r or abs(rr - r) > (p.search_mm + 1.0) * s:
            return None
        c, free_r = cc, rr
    fit_d = 2 * free_r / s
    classes = np.array(p.diameters_mm, dtype=np.float64)
    k = int(np.abs(classes - fit_d).argmin())
    if abs(classes[k] - fit_d) > p.diameter_tol_mm:
        return None
    r_cls = 0.5 * classes[k] * s
    # Final pass: rays and fit with the class radius.
    rad, inner, outer = _ray_edges(g, c[0], c[1], r_cls, p)
    good = ~np.isnan(rad)
    if good.sum() < 0.3 * N_RAYS:
        return None
    pts = c + dirs[good] * rad[good, None]
    keep = np.abs(rad[good] - r_cls) < 2 * tol
    if keep.sum() < 0.3 * N_RAYS:
        return None
    cc = c
    for _ in range(3):
        cc, _ = _fit_circle(pts[keep], r_cls, cc)
        res = np.abs(np.linalg.norm(pts - cc, axis=1) - r_cls)
        keep = res < tol
        if keep.sum() < 0.3 * N_RAYS:
            return None
    coverage = float(keep.sum()) / N_RAYS
    if coverage < p.min_coverage:
        return None
    contrast = float(np.median((outer[good] - inner[good])[keep]))
    rms = float(np.sqrt((res[keep] ** 2).mean())) / s
    center_mm = cc / s - p.origin_mm
    return Circle(center_mm, float(classes[k]), float(fit_d), coverage, rms, contrast)


def find_circles(img: np.ndarray, p: CircleParams, region: np.ndarray | None = None, foreground: np.ndarray | None = None) -> list[Circle]:
    """Circles of the known diameters in the rectified image `img` (BGR or grey).

    `region` (bool, image size): where centres may lie, for example the board.
    `foreground` (bool): pixels that differ from the empty-board reference;
    a circle needs `min_foreground` of its disc there. Best circles first,
    no two overlapping.
    """
    L = lightness(img)
    g, gx, gy = _gradients(L, p.px_per_mm)
    allowed = np.ones(L.shape, bool) if region is None else region.copy()
    if foreground is not None:
        k = int(0.3 * min(p.diameters_mm) * p.px_per_mm) | 1
        allowed &= cv2.dilate(foreground.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    props = _proposals(g, gx, gy, p, allowed, edge_min=0.5 * p.min_contrast / p.px_per_mm)
    found: list[Circle] = []
    for x, y, d in props:
        c = _verify(g, x, y, d, p)
        if c is not None:
            found.append(c)
    if foreground is not None:
        found = [c for c in found if _disc_fraction(foreground, c, p) >= p.min_foreground]
    return _suppress(found)


def _disc_fraction(mask: np.ndarray, c: Circle, p: CircleParams) -> float:
    s = p.px_per_mm
    cx, cy = (c.center_mm + p.origin_mm) * s
    r = 0.5 * c.diameter_mm * s
    x0, x1 = int(max(cx - r, 0)), int(min(cx + r + 1, mask.shape[1]))
    y0, y1 = int(max(cy - r, 0)), int(min(cy + r + 1, mask.shape[0]))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    return float(mask[y0:y1, x0:x1][disc].mean()) if disc.any() else 0.0


def _suppress(found: list[Circle]) -> list[Circle]:
    """Keep the best circle of each overlapping group. Bases can touch, not overlap."""
    found = sorted(found, key=lambda c: (-c.coverage, c.rms_mm))
    keep: list[Circle] = []
    for c in found:
        if any(np.linalg.norm(c.center_mm - k.center_mm) < 0.8 * 0.5 * (c.diameter_mm + k.diameter_mm) for k in keep):
            continue
        keep.append(c)
    return keep
