"""Print sheet with the ChArUco lens calibration target.

    uv run tools/make_charuco.py [--dpi 300]

Writes data/print/charuco.png and .pdf (A4 landscape). Print at 100% or "fit to
page": the lens values (pixels) do not depend on the square size, but the scale must be
the same in both directions. Glue the sheet to something flat and stiff (card, board,
glass). A bent sheet gives a bad calibration.

Then run `uv run battlesight calibrate` (the camera saved by tools/list_cameras.py)
and grab 10 to 20 views: target at the centre, at each edge and corner of the
frame, some tilted by 20 to 40 deg. Use the same camera mode (resolution,
lens, zoom, focus) as for play.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from battlesight.calib import CharucoTarget  # noqa: E402

A4_LANDSCAPE_MM = (297.0, 210.0)


def mm_to_px(mm: float, dpi: float) -> int:
    return int(round(mm / 25.4 * dpi))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dpi", type=float, default=300)
    ap.add_argument("--out", default=str(ROOT / "data" / "print" / "charuco"), help="output path without extension")
    args = ap.parse_args(argv)

    target = CharucoTarget()
    W, H = mm_to_px(A4_LANDSCAPE_MM[0], args.dpi), mm_to_px(A4_LANDSCAPE_MM[1], args.dpi)
    sheet = np.full((H, W), 255, dtype=np.uint8)
    bw, bh = (mm_to_px(n * target.square_mm, args.dpi) for n in target.squares)
    board = target.board().generateImage((bw, bh), marginSize=0, borderBits=1)
    x0, y0 = (W - bw) // 2, mm_to_px(10, args.dpi)
    sheet[y0 : y0 + bh, x0 : x0 + bw] = board

    th = max(1, int(args.dpi / 150))
    fs = args.dpi / 300 * 0.8
    text = (
        f"Battle Sight lens target. ChArUco {target.squares[0]}x{target.squares[1]}, "
        f"square {target.square_mm:g} mm, marker {target.marker_mm:g} mm, DICT_5X5_100. Print at 100% or fit to page, no stretch."
    )
    cv2.putText(sheet, text, (mm_to_px(15, args.dpi), H - mm_to_px(8, args.dpi)), cv2.FONT_HERSHEY_SIMPLEX, fs, 0, th, cv2.LINE_AA)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(sheet)
    img.save(out.with_suffix(".png"), dpi=(args.dpi, args.dpi))
    img.save(out.with_suffix(".pdf"), resolution=args.dpi)
    print(f"wrote {out.with_suffix('.png')} and {out.with_suffix('.pdf')}  ({W}x{H} px at {args.dpi:g} dpi)")
    print(f"check: one square must measure {target.square_mm:g} mm on the print")
    return 0


if __name__ == "__main__":
    sys.exit(main())
