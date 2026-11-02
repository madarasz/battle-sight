"""Print sheet with the four AprilTag 36h11 corner cards.

    uv run tools/make_tags.py [--size-mm 40] [--margin-mm 10] [--dpi 300]

Writes data/tags/corner_tags.png and .pdf (A4). Print at 100% scale, no
"fit to page". Check the 100 mm ruler with a tape measure after printing.
Cut on the grey lines. Put the card's marked corner on the board corner,
card edges flush with the board edges, "TOP" toward the top edge of the board.
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


def draw_card(sheet: np.ndarray, x: int, y: int, tag_id: int, corner: int, size_mm: float, margin_mm: float, dpi: float) -> None:
    size = mm_to_px(size_mm, dpi)
    margin = mm_to_px(margin_mm, dpi)
    card = size + 2 * margin
    # Cut line (light grey) on the card boundary.
    cv2.rectangle(sheet, (x, y), (x + card, y + card), 170, 1)
    sheet[y + margin : y + margin + size, x + margin : x + margin + size] = marker_image(tag_id, size)
    # Mark the corner that goes on the board corner with a small black triangle outside the card.
    offs = {0: (-1, -1), 1: (1, -1), 2: (1, 1), 3: (-1, 1)}[corner]
    cx = x if offs[0] < 0 else x + card
    cy = y if offs[1] < 0 else y + card
    t = mm_to_px(4, dpi)
    tri = np.array([[cx, cy], [cx + offs[0] * t, cy], [cx, cy + offs[1] * t]], dtype=np.int32)
    cv2.fillPoly(sheet, [tri], 0)
    # Labels.
    fs = dpi / 300 * 0.8
    th = max(1, int(dpi / 150))
    cv2.putText(sheet, "TOP", (x + card // 2 - mm_to_px(5, dpi), y - mm_to_px(2, dpi)), cv2.FONT_HERSHEY_SIMPLEX, fs, 0, th, cv2.LINE_AA)
    cv2.putText(sheet, f"id {tag_id}  board corner: {CORNER_NAMES[corner]}", (x, y + card + mm_to_px(6, dpi)), cv2.FONT_HERSHEY_SIMPLEX, fs, 0, th, cv2.LINE_AA)
    cv2.putText(sheet, f"tag {size_mm:g} mm, card {size_mm + 2 * margin_mm:g} mm, black triangle = board corner", (x, y + card + mm_to_px(12, dpi)), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.7, 0, th, cv2.LINE_AA)


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
    ap.add_argument("--dpi", type=float, default=300)
    ap.add_argument("--out", default=str(ROOT / "data" / "tags" / "corner_tags"), help="output path without extension")
    args = ap.parse_args(argv)

    with open(args.config, "rb") as fh:
        tags = tomllib.load(fh)["tags"]
    size_mm = args.size_mm or float(tags["size_mm"])
    margin_mm = args.margin_mm if args.margin_mm is not None else float(tags.get("margin_mm", 0))
    ids = [int(i) for i in tags["ids"]]

    W, H = mm_to_px(A4_MM[0], args.dpi), mm_to_px(A4_MM[1], args.dpi)
    sheet = np.full((H, W), 255, dtype=np.uint8)
    card_mm = size_mm + 2 * margin_mm
    gap_mm = 25.0
    left = (A4_MM[0] - 2 * card_mm - gap_mm) / 2
    top = 25.0
    for i, (tag_id, corner) in enumerate(zip(ids, range(4))):
        col, row = i % 2, i // 2
        x = mm_to_px(left + col * (card_mm + gap_mm), args.dpi)
        y = mm_to_px(top + row * (card_mm + gap_mm + 10), args.dpi)
        draw_card(sheet, x, y, tag_id, corner, size_mm, margin_mm, args.dpi)
    draw_ruler(sheet, mm_to_px(30, args.dpi), mm_to_px(250, args.dpi), 100, args.dpi)
    cv2.putText(sheet, "Battle Sight corner tags. AprilTag 36h11. Print at 100%, no scaling.", (mm_to_px(20, args.dpi), mm_to_px(280, args.dpi)), cv2.FONT_HERSHEY_SIMPLEX, args.dpi / 300 * 0.8, 0, max(1, int(args.dpi / 150)), cv2.LINE_AA)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(sheet)
    img.save(out.with_suffix(".png"), dpi=(args.dpi, args.dpi))
    img.save(out.with_suffix(".pdf"), resolution=args.dpi)
    print(f"wrote {out.with_suffix('.png')} and {out.with_suffix('.pdf')}  ({W}x{H} px at {args.dpi:g} dpi)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
