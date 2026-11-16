"""Position and rotation of light terrain pieces from their outline (lightfit.py)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from battlesight import lightfit
from battlesight.terrain import Piece

PPM = 4.0
SIZE_MM = 260.0
OUTLINE = np.array([[-35, -22], [10, -22], [14, -15], [35, -15], [35, 0], [20, 22], [-20, 22], [-24, 16], [-35, 16]], float)  # 70 x 44 mm, one corner cut, two notches
CARD = (130.0 - 60.0, 130.0 - 52.0, 130.0 + 60.0, 130.0 + 52.0)  # the card rectangle (80 x 64 mm) at (130, 130) plus 20 mm


def board(seed: int = 2) -> np.ndarray:
    """A dark floor with noise, grid lines and a round token."""
    rng = np.random.default_rng(seed)
    n = int(SIZE_MM * PPM)
    img = np.clip(rng.normal(70, 8, (n, n, 3)), 0, 255).astype(np.uint8)
    for k in range(0, n, 100):
        cv2.line(img, (k, 0), (k, n - 1), (95, 95, 95), 2)
        cv2.line(img, (0, k), (n - 1, k), (95, 95, 95), 2)
    cv2.circle(img, (int(60 * PPM), int(200 * PPM)), int(20 * PPM), (60, 90, 200), -1)
    return img


def put(img: np.ndarray, center_mm, angle_deg: float, paint: int = 1, outline: np.ndarray = OUTLINE, ppm: float = PPM) -> np.ndarray:
    """Draw a piece with the outline on the image at a pose (pixel (0, 0) =
    board mm (0, 0)). `paint`: seed of its colour pattern."""
    rng = np.random.default_rng(paint)
    out = img.copy()
    h, w = out.shape[:2]
    small = rng.integers(40, 230, (h // 24 + 1, w // 24 + 1, 3)).astype(np.uint8)
    pattern = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [np.rint(Piece("C1", outline).at(center_mm, angle_deg) * ppm).astype(np.int32)], 255, cv2.LINE_AA)
    out[mask > 127] = pattern[mask > 127]
    return out


def find(img, nominal=(130.0, 130.0), rect=CARD, **over) -> lightfit.Pose:
    return lightfit.locate(img, over.pop("ppm", PPM), (0.0, 0.0), OUTLINE, rect, nominal, 0.0, **over)


def turn(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


@pytest.mark.parametrize("at, angle, paint", [((130.0, 130.0), 0.0, 1), ((136.5, 127.0), 14.0, 2), ((124.0, 133.0), 187.5, 3), ((133.0, 126.0), 352.0, 4)])
def test_finds_the_piece_with_each_paint(at, angle, paint):
    pose = find(put(board(), at, angle, paint))
    assert pose.confirmed and pose.score > 0.6
    assert np.linalg.norm(np.subtract(pose.center_mm, at)) < 1.0
    assert turn(pose.angle_deg, angle) <= 1.0


def test_outline_points_turn_as_the_piece_outline():
    square = np.array([[-35.0, -22.0], [35.0, -22.0], [35.0, 22.0], [-35.0, 22.0]])
    assert Piece("C1", square).at((0.0, 0.0), 90.0)[0] == pytest.approx((22.0, -35.0))  # x right, y down
    pts, cls = lightfit.outline_points(square, 90.0, 1.0)
    assert np.ptp(pts, axis=0) == pytest.approx((44.0, 70.0), abs=1.0) and len(pts) == len(cls) == 228
    # The long sides are vertical now: their normal is along x (class 0).
    long_side = np.abs(np.abs(pts[:, 0]) - 22.0) < 1e-6
    assert set(cls[long_side]) == {0} and set(cls[~long_side]) == {lightfit.BINS // 2}


def test_no_piece_gives_the_nominal_pose_not_confirmed():
    pose = find(board(), nominal=(131.0, 129.0))
    assert not pose.confirmed and pose.center_mm == (131.0, 129.0) and pose.angle_deg == 0.0
    assert pose.score < 0.3


def test_another_piece_is_not_confirmed():
    disc = np.array([[30 * np.cos(a), 30 * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 40, endpoint=False)])
    assert not find(put(board(), (130.0, 130.0), 0.0, outline=disc)).confirmed


def test_card_rectangle_limits_the_search():
    assert lightfit.origin_box(OUTLINE, 0.0, (0.0, 0.0, 100.0, 60.0)) == pytest.approx((35.0, 22.0, 65.0, 38.0))
    assert lightfit.origin_box(OUTLINE, 90.0, (0.0, 0.0, 100.0, 60.0)) is None  # 70 mm does not fit in 60
    img = put(board(), (190.0, 60.0), 0.0)  # the piece is out of the card rectangle
    assert not find(img).confirmed
    wide = (CARD[0], 0.0, 250.0, CARD[3])
    pose = find(img, rect=wide)
    assert pose.confirmed and np.linalg.norm(np.subtract(pose.center_mm, (190.0, 60.0))) < 1.0
    small = (100.0, 110.0, 160.0, 150.0)  # the outline fits at no angle
    assert not find(img, rect=small).confirmed


def test_piece_partly_covered_is_found():
    img = put(board(), (132.0, 128.0), 20.0)
    cv2.circle(img, (int(150 * PPM), int(140 * PPM)), int(14 * PPM), (20, 20, 20), -1)  # a base over part of the piece
    pose = find(img)
    assert pose.confirmed and np.linalg.norm(np.subtract(pose.center_mm, (132.0, 128.0))) < 1.5


def test_low_resolution_image():
    ppm = 1.4
    img = cv2.resize(put(board(), (128.0, 133.0), 33.0), None, fx=ppm / PPM, fy=ppm / PPM, interpolation=cv2.INTER_AREA)
    pose = find(img, ppm=ppm)
    assert pose.confirmed and np.linalg.norm(np.subtract(pose.center_mm, (128.0, 133.0))) < 1.5 and turn(pose.angle_deg, 33.0) <= 2.0
