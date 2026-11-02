"""Print sheet with the four AprilTag 36h11 corner cards.

    uv run tools/make_tags.py [--size-mm 40] [--margin-mm 5] [--guide-mm 40] [--dpi 300]

Writes data/tags/corner_tags.png and .pdf (A4, all four cards on one page). Print at 100% scale, no "fit to page". Check the 100 mm ruler with a
tape measure after printing. Cut on the grey lines.

Each card is a square on the table at its board corner. The tag sits in the
outer quadrant, diagonal to the corner. A black L in the inner quadrant
marks the board corner. Rest the board corner on the card with both board
edges against the inner side of the L, lines still visible. The card lies under the board, so nothing covers the play
area. The tag's inner margin corner is then the board corner.

The white margin is the quiet zone the detector needs around the black
border: at least one module (size / 8). Arm length does not change accuracy:
the board corner is the tag's inner margin corner, whatever the card's
rotation. The L only holds that corner in place.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CORNER_NAMES = ("top-left", "top-right", "bottom-right", "bottom-left")
A4_MM = (210.0, 297.0)


def mm_to_px(mm: float, dpi: float) -> int:
    return int(round(mm / 25.4 * dpi))


def marker_image(tag_id: int, side_px: int) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    base = cv2.aruco.generateImageMarker(dictionary, tag_id, 8 * 16)  # 16 px per module
    return cv2.resize(base, (side_px, side_px), interpolation=cv2.INTER_NEAREST)


# Unit vectors that point from each board corner into the board (x right, y down).
INWARD = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=np.float64)


def draw_card(sheet: np.ndarray, corner_mm: np.ndarray, tag_id: int, corner: int, size_mm: float, margin_mm: float, guide_mm: float, dpi: float) -> None:
    """Draw one corner card. `corner_mm` is the board corner position on the page.

    The card is a square. The tag sits in the outer quadrant, diagonal to the
    board corner. A black L marks the board corner and the first `guide_mm`
    of both board edges. The board corner rests on the inner quadrant with
    its edges against the inner side of the L, lines still visible. The card
    lies under the board, so nothing covers the play area.
    """
    ix, iy = INWARD[corner]
    card_mm = size_mm + 2 * margin_mm

    def px(p) -> tuple[int, int]:
        return (mm_to_px(p[0], dpi), mm_to_px(p[1], dpi))

    def rect(a, b) -> tuple[tuple[int, int], tuple[int, int]]:
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        return px(lo), px(hi)

    P = np.asarray(corner_mm, dtype=np.float64)
    inward = np.array([ix, iy])
    # Cut line (light grey): one square, tag quadrant plus board quadrant.
    cv2.rectangle(sheet, *rect(P - inward * card_mm, P + inward * guide_mm), 170, 1)

    tag0 = px(P - inward * margin_mm - (inward > 0) * size_mm)  # tag TL corner on the page
    side = mm_to_px(size_mm, dpi)
    sheet[tag0[1] : tag0[1] + side, tag0[0] : tag0[0] + side] = marker_image(tag_id, side)

    # Board corner L. The lines lie just outside the board edges and stay
    # visible: board too far out covers a line, too far in leaves a white gap.
    # They meet only at the corner point, so the tag's white margin stays clean.
    lw = 0.8
    cv2.rectangle(sheet, *rect(P, P + np.array([ix * guide_mm, -iy * lw])), 0, -1)
    cv2.rectangle(sheet, *rect(P, P + np.array([-ix * lw, iy * guide_mm])), 0, -1)

    # Labels in the free quadrant beside the tag, and in the board quadrant.
    fs = dpi / 300 * 0.7
    th = max(1, int(dpi / 150))
    font = cv2.FONT_HERSHEY_SIMPLEX
    free = P + np.array([ix * guide_mm / 2, -iy * card_mm / 2])
    for i, text in enumerate((f"id {tag_id}", CORNER_NAMES[corner])):
        (tw, tht), _ = cv2.getTextSize(text, font, fs, th)
        o = px(free + np.array([0, (i - 0.5) * 6]))
        cv2.putText(sheet, text, (o[0] - tw // 2, o[1] + tht // 2), font, fs, 0, th, cv2.LINE_AA)
    (tw, tht), _ = cv2.getTextSize("BOARD", font, fs, th)
    o = px(P + inward * guide_mm / 2)
    cv2.putText(sheet, "BOARD", (o[0] - tw // 2, o[1] + tht // 2), font, fs, 120, th, cv2.LINE_AA)


def draw_ruler(sheet: np.ndarray, x: int, y: int, length_mm: float, dpi: float) -> None:
    th = max(1, int(dpi / 150))
    cv2.line(sheet, (x, y), (x + mm_to_px(length_mm, dpi), y), 0, th)
    for mm in range(0, int(length_mm) + 1, 10):
        px = x + mm_to_px(mm, dpi)
        cv2.line(sheet, (px, y), (px, y - mm_to_px(4, dpi)), 0, th)
        cv2.putText(sheet, str(mm), (px - mm_to_px(2, dpi), y + mm_to_px(5, dpi)), cv2.FONT_HERSHEY_SIMPLEX, dpi / 300 * 0.6, 0, th, cv2.LINE_AA)
    cv2.putText(sheet, f"{length_mm:g} mm check ruler. Measure after printing.", (x, y - mm_to_px(7, dpi)), cv2.FONT_HERSHEY_SIMPLEX, dpi / 300 * 0.8, 0, th, cv2.LINE_AA)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "data" / "board.toml"))
    ap.add_argument("--size-mm", type=float, default=None, help="black square size (default from config)")
    ap.add_argument("--margin-mm", type=float, default=None, help="white margin (default from config)")
    ap.add_argument("--guide-mm", type=float, default=None, help="guide end length along the board edge (default from config)")
    ap.add_argument("--dpi", type=float, default=300)
    ap.add_argument("--out", default=str(ROOT / "data" / "tags" / "corner_tags"), help="output path without extension")
    args = ap.parse_args(argv)

    with open(args.config, "rb") as fh:
        tags = tomllib.load(fh)["tags"]
    size_mm = args.size_mm or float(tags["size_mm"])
    margin_mm = args.margin_mm if args.margin_mm is not None else float(tags.get("margin_mm", 0))
    guide_mm = args.guide_mm or float(tags.get("guide_mm", 100.0))
    ids = [int(i) for i in tags["ids"]]

    W, H = mm_to_px(A4_MM[0], args.dpi), mm_to_px(A4_MM[1], args.dpi)
    sheet = np.full((H, W), 255, dtype=np.uint8)
    card_mm = size_mm + 2 * margin_mm
    gap_mm = 5.0
    # Two rows of two square cards, each in the orientation it has on the
    # table, board quadrants toward the middle of the page.
    row_mm = 2 * (card_mm + guide_mm) + gap_mm
    if row_mm > A4_MM[0] - 10:
        raise SystemExit(f"cards too big for one A4 page: {row_mm:g} mm wide, max {A4_MM[0] - 10:g} mm. Use a shorter --guide-mm.")
    left, top = (A4_MM[0] - row_mm) / 2, 20.0
    x_l, x_r = left + card_mm, left + row_mm - card_mm
    y_t, y_b = top + card_mm, top + row_mm - card_mm
    corners = np.array([[x_l, y_t], [x_r, y_t], [x_r, y_b], [x_l, y_b]])
    for corner, tag_id in enumerate(ids):
        draw_card(sheet, corners[corner], tag_id, corner, size_mm, margin_mm, guide_mm, args.dpi)

    th = max(1, int(args.dpi / 150))
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = args.dpi / 300 * 0.8
    lines = [
        "Cut on the grey lines. Put each card on the table at its board corner.",
        "Rest the board corner on the card. Board edges touch the inner side of the black L.",
        "Both lines stay fully visible, with no white gap between line and board.",
        "The tag stays outside the board, diagonal to the corner.",
        f"tag {size_mm:g} mm, white margin {margin_mm:g} mm (do not trim), card {card_mm + guide_mm:g} x {card_mm + guide_mm:g} mm",
    ]
    for i, text in enumerate(lines):
        cv2.putText(sheet, text, (mm_to_px(15, args.dpi), mm_to_px(top + row_mm + 15 + i * 7, args.dpi)), font, fs, 0, th, cv2.LINE_AA)
    draw_ruler(sheet, mm_to_px(30, args.dpi), mm_to_px(265, args.dpi), 100, args.dpi)
    cv2.putText(sheet, "Battle Sight corner cards. AprilTag 36h11. Print at 100%, no scaling.", (mm_to_px(15, args.dpi), mm_to_px(287, args.dpi)), font, fs, 0, th, cv2.LINE_AA)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(sheet)
    img.save(out.with_suffix(".png"), dpi=(args.dpi, args.dpi))
    img.save(out.with_suffix(".pdf"), resolution=args.dpi)
    print(f"wrote {out.with_suffix('.png')} and {out.with_suffix('.pdf')}  ({W}x{H} px at {args.dpi:g} dpi)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
