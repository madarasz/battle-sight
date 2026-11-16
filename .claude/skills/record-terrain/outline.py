"""Helper of the skill record-terrain: look at a photo with pixel coordinates,
and turn an outline in photo pixels into a piece file.

    uv run .claude/skills/record-terrain/outline.py view photo.jpg --out view.png [--points points.json] [--box X0 Y0 X1 Y1] [--no-grid]
    uv run .claude/skills/record-terrain/outline.py suggest photo.jpg --out points.json
    uv run .claude/skills/record-terrain/outline.py simplify points.json --length 62 --out points.json [--tolerance-mm 0.2]
    uv run .claude/skills/record-terrain/outline.py save points.json --key C2 --length 62 --width 46 [--source photo.jpg] [--terrain tomb-world] [--check check.png]

The points file is a JSON list of [x, y] in photo pixels, in order along
the outline. The photo shows the underside of the piece, so `save` mirrors
the outline: the piece file holds the outline as seen from above.

Exit codes: 0 ok, 1 no frame angle agrees with the measured sizes, 2 bad input.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from battlesight import terrain as ter

VIEW_PX = 1600  # long side of the view image
GRID_LINES = 12  # about this many grid lines on the long side
STEPS = (5, 10, 20, 25, 50, 100, 200, 250, 500)  # grid pitch in photo pixels
WARN_RATIO, MAX_RATIO = 0.03, 0.08  # length scale against width scale
MIN_AREA_PART = 0.005  # a smaller dark area is not the piece (part of the photo)
STEP_PART = 0.004  # `suggest`: the outline follows the mask within this part of the piece size
CHECK_PX_PER_MM = 10.0
ANGLE_STEP_DEG = 0.25  # frame angles that `to_piece` tries
SAME_RATIO = 0.003  # frame angles agree equally with the sizes within this
MAX_TURN_DEG = 20.0  # the measured length is along a photo side within this
SIMPLIFY_MM = 0.2  # `simplify`: default tolerance


class OutlineError(ValueError):
    pass


def load_points(path: str | Path) -> np.ndarray:
    try:
        pts = np.asarray(json.loads(Path(path).read_text()), float)
    except (OSError, ValueError) as exc:
        raise OutlineError(f"cannot read {path}: {exc}") from exc
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        raise OutlineError(f"{path}: must be a list of 3 or more points [x, y]")
    return pts


def view(photo: np.ndarray, points: np.ndarray | None = None, box=None, grid: bool = True) -> np.ndarray:
    """The part `box` (x0, y0, x1, y1 in photo pixels; None = all) of the
    photo, made large, with a grid in photo pixels (`grid` False: only
    marks at the top and left edge) and the outline with its point numbers."""
    h, w = photo.shape[:2]
    x0, y0, x1, y1 = (0, 0, w, h) if box is None else box
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise OutlineError("the box is empty")
    s = VIEW_PX / max(x1 - x0, y1 - y0)
    img = cv2.resize(photo[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    step = next((v for v in STEPS if max(x1 - x0, y1 - y0) / v <= GRID_LINES), STEPS[-1])

    def text(label: str, at: tuple[int, int], color) -> None:
        cv2.putText(img, label, at, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, label, at, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    for gx in range(math.ceil(x0 / step) * step, x1, step):
        x = round((gx - x0) * s)
        cv2.line(img, (x, 0), (x, img.shape[0] if grid else 10), (255, 200, 0), 1)
        text(str(gx), (x + 3, 14), (255, 200, 0))
    for gy in range(math.ceil(y0 / step) * step, y1, step):
        y = round((gy - y0) * s)
        cv2.line(img, (0, y), (img.shape[1] if grid else 10, y), (255, 200, 0), 1)
        text(str(gy), (3, y - 4), (255, 200, 0))
    if points is not None:
        px = np.rint((points - (x0, y0)) * s).astype(np.int32)
        cv2.polylines(img, [px.reshape(-1, 1, 2)], True, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.polylines(img, [px.reshape(-1, 1, 2)], True, (0, 255, 255), 1, cv2.LINE_AA)
        for k, (x, y) in enumerate(px):
            cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1, cv2.LINE_AA)
            text(str(k), (int(x) + 5, int(y) - 5), (0, 0, 255))
    return img


def suggest(photo: np.ndarray) -> np.ndarray:
    """A first outline (N, 2) in photo pixels: the largest area that is not
    the table and does not touch the photo edge. It can include a shadow or
    the side of the piece: the agent corrects it."""
    lab = cv2.cvtColor(photo, cv2.COLOR_BGR2LAB).astype(np.float32)
    table = np.median(lab.reshape(-1, 3), axis=0)
    diff = np.clip(np.linalg.norm(lab - table, axis=2), 0, 255).astype(np.uint8)
    h, w = diff.shape
    _, mask = cv2.threshold(cv2.GaussianBlur(diff, (0, 0), max(1.0, 0.002 * max(h, w))), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = 2 * max(1, round(0.004 * max(h, w))) + 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    best, best_area = None, MIN_AREA_PART * h * w
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if x <= 0 or y <= 0 or x + cw >= w or y + ch >= h:
            continue
        if cv2.contourArea(c) > best_area:
            best, best_area = c, cv2.contourArea(c)
    if best is None:
        raise OutlineError("no piece found: trace the outline by eye")
    return cv2.approxPolyDP(best, STEP_PART * math.sqrt(best_area), True).reshape(-1, 2).astype(float)


def simplify(points_px: np.ndarray, length_mm: float, tolerance_mm: float = SIMPLIFY_MM) -> np.ndarray:
    """The outline with fewer points: each removed point was within
    `tolerance_mm` of the new line (Douglas-Peucker). The scale comes from
    the long side of the outline box and `length_mm`; the point order stays."""
    if length_mm <= 0 or tolerance_mm < 0:
        raise OutlineError("length must be more than 0 and tolerance 0 or more")
    px_per_mm = float(np.ptp(points_px, axis=0).max()) / length_mm
    pts = np.asarray(points_px, np.float32).reshape(-1, 1, 2)
    return cv2.approxPolyDP(pts, tolerance_mm * px_per_mm, True).reshape(-1, 2).astype(float)


def to_piece(points_px: np.ndarray, length_mm: float, width_mm: float) -> tuple[np.ndarray, float, float, float]:
    """The outline (N, 2) in mm in the piece frame, as seen from above, from
    an outline in pixels of a photo of the underside.

    The frame angle is the one, within `MAX_TURN_DEG` of the photo axes,
    where the sizes of the outline along and across the frame agree best
    with length : width (a minimum area rectangle is not usable: one long
    slanted edge turns it away from the direction in which a person
    measures). Of the angles that agree equally, the one nearest to the
    photo axes. One scale for both axes (the mean of the length scale and
    the width scale), and x changes sign: the underside is the mirror image.

    Also returns: the difference of the two scales as a part of 1 (camera
    tilt, a wrong size, a bad outline), the frame angle in degrees from the
    nearest photo axis, and the scale range of all angles that agree
    equally, as a part of 1 (how sure the scale is)."""
    if length_mm <= 0 or width_mm <= 0 or width_mm > length_mm:
        raise OutlineError("length and width must be more than 0, and the length is the longer one")
    pts = np.asarray(points_px, float).reshape(-1, 2)

    def turned(deg: float) -> np.ndarray:
        a = math.radians(deg)
        return pts @ np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]]).T

    tries = []  # (ratio, off, deg, scale)
    for deg in np.arange(-90.0, 90.0, ANGLE_STEP_DEG):
        off = abs((deg + 45.0) % 90.0 - 45.0)  # from the nearest photo axis
        if off > MAX_TURN_DEG:
            continue
        size = np.ptp(turned(deg), axis=0)
        if size.min() <= 0:
            raise OutlineError("the outline has no area")
        scales = np.array([length_mm, width_mm]) / size
        tries.append((abs(scales[0] / scales[1] - 1.0), off, deg, float(scales.mean())))
    least = min(t[0] for t in tries)
    good = [t for t in tries if t[0] <= least + SAME_RATIO]
    ratio, off, deg, scale = min(good, key=lambda t: (round(t[0], 3), t[1]))
    spread = (max(t[3] for t in good) - min(t[3] for t in good)) / scale
    return ter.piece_frame(turned(deg) * scale * (-1.0, 1.0)), float(ratio), float(off), float(spread)


def check_image(outline_mm: np.ndarray, key: str) -> np.ndarray:
    """The piece outline as seen from above, with a 10 mm grid."""
    half = np.abs(outline_mm).max(axis=0) + 8.0
    size = np.ceil(2 * half * CHECK_PX_PER_MM).astype(int)
    img = np.full((size[1], size[0], 3), 255, np.uint8)

    def px(mm):
        return np.rint((np.asarray(mm, float) + half) * CHECK_PX_PER_MM).astype(np.int32)

    for v in np.arange(-math.floor(half.max() / 10) * 10, half.max(), 10.0):
        color = (150, 150, 150) if v == 0 else (225, 225, 225)
        cv2.line(img, tuple(px([v, -half[1]])), tuple(px([v, half[1]])), color, 1)
        cv2.line(img, tuple(px([-half[0], v])), tuple(px([half[0], v])), color, 1)
    cv2.polylines(img, [px(outline_mm).reshape(-1, 1, 2)], True, (0, 0, 0), 2, cv2.LINE_AA)
    w, h = np.ptp(outline_mm, axis=0)
    cv2.putText(img, f"{key} from above, {w:.1f} x {h:.1f} mm, grid 10 mm", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 1, cv2.LINE_AA)
    return img


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Helper of the skill record-terrain.")
    sub = p.add_subparsers(dest="command", required=True)
    v = sub.add_parser("view", help="photo with a pixel grid, and the outline")
    v.add_argument("photo")
    v.add_argument("--out", required=True)
    v.add_argument("--points")
    v.add_argument("--box", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"))
    v.add_argument("--no-grid", action="store_true", help="only marks at the edge: thin rim edges stay in view")
    g = sub.add_parser("suggest", help="first outline from the dark area on the table")
    g.add_argument("photo")
    g.add_argument("--out", required=True)
    m = sub.add_parser("simplify", help="fewer points: remove steps smaller than a tolerance")
    m.add_argument("points")
    m.add_argument("--length", type=float, required=True, metavar="MM", help="longest size of the footprint, for the scale")
    m.add_argument("--tolerance-mm", type=float, default=SIMPLIFY_MM)
    m.add_argument("--out", required=True)
    s = sub.add_parser("save", help="write the piece file")
    s.add_argument("points")
    s.add_argument("--key", required=True)
    s.add_argument("--length", type=float, required=True, metavar="MM", help="longest size of the footprint")
    s.add_argument("--width", type=float, required=True, metavar="MM", help="size at a right angle to the length")
    s.add_argument("--source", default="", help="photo file name, for the piece file")
    s.add_argument("--terrain", default="tomb-world")
    s.add_argument("--root", default=str(ter.DEFAULT_PIECES_ROOT))
    s.add_argument("--check", help="write an image of the outline as seen from above")
    args = p.parse_args(argv)
    try:
        if args.command == "simplify":
            pts = load_points(args.points)
            out = simplify(pts, args.length, args.tolerance_mm)
            Path(args.out).write_text(json.dumps(np.round(out, 1).tolist()))
            print(f"wrote {args.out}: {len(pts)} -> {len(out)} points")
            return 0
        if args.command == "save":
            outline, ratio, off, spread = to_piece(load_points(args.points), args.length, args.width)
            if ratio > MAX_RATIO:
                print(f"error: length and width give scales that differ by {100 * ratio:.0f} %: check the outline, the sizes and the camera tilt", file=sys.stderr)
                return 1
            if ratio > WARN_RATIO:
                print(f"warning: length and width give scales that differ by {100 * ratio:.0f} %")
            path = Path(args.root) / args.terrain / f"{args.key}.toml"
            path.parent.mkdir(parents=True, exist_ok=True)
            ter.write_piece(path, ter.Piece(args.key, outline), args.source)
            if args.check:
                cv2.imwrite(args.check, check_image(outline, args.key))
            w, h = np.ptp(outline, axis=0)
            print(f"saved {path}: {len(outline)} points, {w:.1f} x {h:.1f} mm, scale difference {100 * ratio:.1f} %, frame {off:.1f} degrees from the photo axes, scale sure within {100 * spread:.1f} %")
            return 0
        photo = cv2.imread(args.photo)
        if photo is None:
            raise OutlineError(f"cannot read {args.photo}")
        if args.command == "suggest":
            pts = suggest(photo)
            Path(args.out).write_text(json.dumps(np.round(pts, 1).tolist()))
            print(f"wrote {args.out}: {len(pts)} points")
            return 0
        cv2.imwrite(args.out, view(photo, load_points(args.points) if args.points else None, args.box, not args.no_grid))
        print(f"wrote {args.out}")
        return 0
    except OutlineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
