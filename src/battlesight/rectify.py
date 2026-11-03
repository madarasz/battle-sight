"""Board plane geometry: homography from the corner tags, board thickness
correction, top-down warp, and checks of the result.

Priority 2 of the plan. All functions work on one frame. No camera, no window.

Coordinates:

- board mm: origin at the top-left board corner, x right, y down, on the play
  surface (`board.thickness_mm` above the table).
- image px: pixels of the camera frame, with lens distortion.
- rectified px: pixels of the top-down image, `rectify.px_per_mm` per mm,
  board origin at (`rectify.margin_mm`, `rectify.margin_mm`) mm.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import cv2
import numpy as np

from .calib import Lens, TagObs
from .config import BoardConfig

EDGE_NAMES = ("top", "right", "bottom", "left")


@dataclass
class BoardView:
    """Maps between board mm and image pixels for one frame."""

    H: np.ndarray  # (3, 3) board surface mm -> undistorted image px
    lens: Lens
    camera_mm: np.ndarray  # (3,) camera centre: x, y in board mm, height above the table in mm
    surface_scale: float  # table-plane scale of a surface point about the camera nadir (>= 1)
    reproj_rms_mm: float  # fit error of the 16 tag corners, in table mm
    reproj_max_mm: float

    def mm_to_px(self, pts_mm: np.ndarray) -> np.ndarray:
        """Board surface mm (N, 2) -> image px (N, 2)."""
        pts = np.asarray(pts_mm, dtype=np.float64).reshape(-1, 1, 2)
        und = cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)
        return self.lens.distort_px(und)

    def px_to_mm(self, pts_px: np.ndarray) -> np.ndarray:
        """Image px (N, 2) -> board surface mm (N, 2)."""
        und = self.lens.undistort_px(pts_px).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(und, np.linalg.inv(self.H)).reshape(-1, 2)


def fit_view(cfg: BoardConfig, obs: dict[int, TagObs], lens: Lens) -> BoardView:
    """Fit the board plane to the four corner tags.

    1. Homography table mm -> undistorted px from all 16 tag corners (least squares).
    2. Camera pose from the same points (planar PnP) gives the camera centre.
    3. A point on the play surface, `thickness_mm` above the table, shows in
       the image where the camera ray through it meets the table: further out
       from the nadir by the factor h / (h - t). The surface homography is the
       table homography after that scale.
    """
    missing = [i for i in cfg.tag_ids if i not in obs]
    if missing:
        raise ValueError(f"need all four tags, missing {missing}")
    tag_mm = cfg.tag_corners_mm()
    mm = np.concatenate([tag_mm[i] for i in cfg.tag_ids])
    und = lens.undistort_px(np.concatenate([obs[i].corners for i in cfg.tag_ids]))

    H_table, _ = cv2.findHomography(mm, und, 0)
    if H_table is None:
        raise ValueError("homography fit failed")
    back = cv2.perspectiveTransform(und.reshape(-1, 1, 2), np.linalg.inv(H_table)).reshape(-1, 2)
    err = np.linalg.norm(back - mm, axis=1)

    obj = np.c_[mm, np.zeros(len(mm))]
    ok, rvec, tvec = cv2.solvePnP(obj, und, lens.K, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise ValueError("camera pose fit failed")
    R, _ = cv2.Rodrigues(rvec)
    C = (-R.T @ tvec).ravel()
    # Board z axis = x cross y points into the table, so the camera has z < 0.
    height = -C[2]
    t = cfg.thickness_mm
    if height <= max(2.0 * t, 1.0):
        raise ValueError(f"camera height {height:.0f} mm is not above the board")
    s = height / (height - t)
    S = np.array([[s, 0, (1 - s) * C[0]], [0, s, (1 - s) * C[1]], [0, 0, 1]], dtype=np.float64)
    H = H_table @ S
    return BoardView(H / H[2, 2], lens, np.array([C[0], C[1], height]), float(s), float(np.sqrt((err**2).mean())), float(err.max()))


def swapped_fits_better(cfg: BoardConfig, obs: dict[int, TagObs], lens: Lens, current: BoardView) -> bool:
    """True when the tags fit much better with board width and height swapped:
    the usual cause of a bad fit is a config with the edges the wrong way round."""
    other = dataclasses.replace(cfg, width_mm=cfg.height_mm, height_mm=cfg.width_mm)
    try:
        alt = fit_view(other, obs, lens)
    except ValueError:
        return False
    return alt.reproj_max_mm <= cfg.max_reproj_mm and alt.reproj_max_mm < 0.5 * current.reproj_max_mm


# --- top-down image ----------------------------------------------------------


def rect_transform(cfg: BoardConfig) -> tuple[np.ndarray, tuple[int, int]]:
    """(3x3 board mm -> rectified px, (width, height) of the rectified image)."""
    p, m = cfg.rectify_px_per_mm, cfg.rectify_margin_mm
    A = np.array([[p, 0, m * p], [0, p, m * p], [0, 0, 1]], dtype=np.float64)
    size = (int(round((cfg.width_mm + 2 * m) * p)), int(round((cfg.height_mm + 2 * m) * p)))
    return A, size


def warp_board(frame: np.ndarray, view: BoardView, cfg: BoardConfig) -> np.ndarray:
    """Top-down image of the board at `rectify.px_per_mm`, lens distortion removed."""
    A, size = rect_transform(cfg)
    if not view.lens.has_distortion:
        return cv2.warpPerspective(frame, view.H @ np.linalg.inv(A), size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=(0, 0, 0))
    # One remap for distortion and homography together: rectified px -> mm -> image px.
    xs, ys = np.meshgrid(np.arange(size[0], dtype=np.float64), np.arange(size[1], dtype=np.float64))
    mm = (np.c_[xs.ravel(), ys.ravel()] - A[:2, 2]) / cfg.rectify_px_per_mm
    px = view.mm_to_px(mm).reshape(size[1], size[0], 2).astype(np.float32)
    return cv2.remap(frame, px[..., 0], px[..., 1], cv2.INTER_LINEAR, borderValue=(0, 0, 0))


def mm_to_rect(cfg: BoardConfig, pts_mm: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_mm, dtype=np.float64).reshape(-1, 2)
    return (pts + cfg.rectify_margin_mm) * cfg.rectify_px_per_mm


# --- checks on the top-down image --------------------------------------------


@dataclass
class EdgeFit:
    """A board edge found in the rectified image.

    `offset_mm` is the distance of the found edge from its nominal line at the
    edge midpoint, positive into the board. `slope` is the tilt of the found
    edge in board mm: dx/dy for left and right, dy/dx for top and bottom.
    """

    offset_mm: float
    slope: float
    n: int
    rms_mm: float

    @property
    def angle_deg(self) -> float:
        return float(np.degrees(np.arctan(self.slope)))


def _gray(rect: np.ndarray) -> np.ndarray:
    g = rect if rect.ndim == 2 else cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.0)


def _profiles(g: np.ndarray, cfg: BoardConfig, vertical: bool, pos_mm: float, along_mm: np.ndarray, half_mm: float) -> tuple[np.ndarray, np.ndarray]:
    """Grey profiles across a line. Vertical line x = pos_mm: one row per
    sample y in along_mm, columns from pos - half to pos + half. Returns
    (profiles (n, k), offsets in mm (k,))."""
    p, m = cfg.rectify_px_per_mm, cfg.rectify_margin_mm
    k = int(round(2 * half_mm * p)) + 1
    offs = np.linspace(-half_mm, half_mm, k)
    across = np.rint((pos_mm + offs + m) * p).astype(int)
    along = np.rint((along_mm + m) * p).astype(int)
    if vertical:
        rows, cols = along[:, None], across[None, :]
    else:
        rows, cols = across[None, :], along[:, None]
    rows = np.clip(rows, 0, g.shape[0] - 1)
    cols = np.clip(cols, 0, g.shape[1] - 1)
    return g[rows, cols], offs


def _subpixel(v: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Parabola peak offset (in samples) around index j of each row of v."""
    i = np.arange(len(v))
    j = np.clip(j, 1, v.shape[1] - 2)
    a, b, c = v[i, j - 1], v[i, j], v[i, j + 1]
    den = a - 2 * b + c
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(np.abs(den) > 1e-9, 0.5 * (a - c) / den, 0.0)
    return j + np.clip(d, -0.5, 0.5)


def _robust_line(t: np.ndarray, v: np.ndarray, min_n: int) -> tuple[float, float, int, float] | None:
    """Fit v = a + b * (t - mid), dropping outliers. Returns (a, b, n, rms) or None."""
    if len(t) < min_n:
        return None
    mid = 0.5 * (t.min() + t.max())
    keep = np.abs(v - np.median(v)) < 3.0
    for _ in range(4):
        if keep.sum() < min_n:
            return None
        b, a = np.polyfit(t[keep] - mid, v[keep], 1)
        r = v - (a + b * (t - mid))
        mad = np.median(np.abs(r[keep])) * 1.4826
        keep = np.abs(r) < max(0.75, 2.5 * mad)
    if keep.sum() < min_n:
        return None
    r = r[keep]
    return float(a), float(b), int(keep.sum()), float(np.sqrt((r**2).mean()))


def measure_edges(rect: np.ndarray, cfg: BoardConfig, search_mm: float = 8.0, corner_skip_mm: float = 50.0, step_mm: float = 4.0) -> dict[str, EdgeFit | None]:
    """Find the four board edges in the rectified image.

    Samples the strongest brightness step across each nominal edge, away from
    the corners (the cards and the L lines lie there), and fits a line with
    outlier rejection (walls and models over the edge). None for an edge with
    too few clear samples.
    """
    g = _gray(rect)
    W, H = cfg.width_mm, cfg.height_mm
    out: dict[str, EdgeFit | None] = {}
    for name in EDGE_NAMES:
        vertical = name in ("left", "right")
        pos = {"top": 0.0, "bottom": H, "left": 0.0, "right": W}[name]
        inward = 1.0 if name in ("top", "left") else -1.0
        length = H if vertical else W
        along = np.arange(corner_skip_mm, length - corner_skip_mm, step_mm)
        prof, offs = _profiles(g, cfg, vertical, pos, along, search_mm)
        grad = np.abs(np.gradient(prof, axis=1))
        j = grad.argmax(axis=1)
        peak = grad[np.arange(len(j)), j]
        good = peak > np.maximum(3.0, 4.0 * np.median(grad, axis=1))
        step = offs[1] - offs[0]
        found = offs[0] + _subpixel(grad, j) * step
        fit = _robust_line(along[good], inward * found[good], min_n=10)
        out[name] = None if fit is None else EdgeFit(fit[0], fit[1] * inward, fit[2], fit[3])
    return out


def corner_angles_deg(edges: dict[str, EdgeFit | None]) -> list[float | None]:
    """Interior board corner angles TL, TR, BR, BL from the fitted edges."""
    # Direction of each edge as (dx, dy) in board mm, going right (top, bottom) or down (left, right).
    d = {}
    for name, e in edges.items():
        if e is None:
            continue
        b = e.slope
        d[name] = np.array([1.0, b]) if name in ("top", "bottom") else np.array([b, 1.0])
    pairs = [("top", "left", 1, 1), ("top", "right", -1, 1), ("bottom", "right", -1, -1), ("bottom", "left", 1, -1)]
    out: list[float | None] = []
    for h, v, sx, sy in pairs:
        if h not in d or v not in d:
            out.append(None)
            continue
        a, b = d[h] * sx, d[v] * sy  # both point away from the corner
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        out.append(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))))
    return out


def measure_grid_lines(rect: np.ndarray, cfg: BoardConfig, search_mm: float = 4.0, step_mm: float = 2.0, min_contrast: float = 10.0) -> dict[str, list[float | None]]:
    """Find the printed grid lines (dark lines on the board) near their
    nominal positions from `cfg.grid_lines_mm()`.

    Returns {"x": offsets of the vertical lines, "y": offsets of the
    horizontal lines}, each the median offset in mm (positive = larger
    coordinate), or None for a line hidden too much.
    """
    g = _gray(rect)
    xs, ys = cfg.grid_lines_mm()
    m = cfg.grid_margin_mm
    out: dict[str, list[float | None]] = {}
    for key, lines, length, vertical in (("x", xs, cfg.height_mm, True), ("y", ys, cfg.width_mm, False)):
        along = np.arange(m, length - m, step_mm)
        res: list[float | None] = []
        for pos in lines:
            prof, offs = _profiles(g, cfg, vertical, float(pos), along, search_mm)
            j = prof.argmin(axis=1)
            low = prof[np.arange(len(j)), j]
            good = (np.median(prof, axis=1) - low > min_contrast) & (j > 0) & (j < prof.shape[1] - 1)
            if good.sum() < 10:
                res.append(None)
                continue
            found = offs[0] + _subpixel(prof[good], j[good]) * (offs[1] - offs[0])
            res.append(float(np.median(found)))
        out[key] = res
    return out


def grid_spacing_errors_mm(cfg: BoardConfig, offsets: dict[str, list[float | None]]) -> list[float]:
    """Spacing error of each pair of neighbouring found grid lines, in mm."""
    out = []
    for key, nominal in zip(("x", "y"), cfg.grid_lines_mm()):
        o = offsets.get(key, [])
        for k in range(len(nominal) - 1):
            if k + 1 < len(o) and o[k] is not None and o[k + 1] is not None:
                out.append(float(o[k + 1] - o[k]))
    return out
