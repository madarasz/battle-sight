"""Synthetic board images for tests. No camera needed."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from battlesight.config import BoardConfig

BOARD_GRAY = 70  # dark board, like Tomb World
LINE_GRAY = 30  # printed grid lines on the board


def make_config(**over) -> BoardConfig:
    base = dict(
        name="test",
        width_mm=606.0,
        height_mm=703.0,
        tag_ids=(0, 1, 2, 3),
        tag_size_mm=40.0,
        tag_margin_mm=5.0,
        drift_tolerance_mm=2.0,
        lost_after_frames=3,
        settle_frames=5,
    )
    base.update(over)
    return BoardConfig(**base)


def render_board(cfg: BoardConfig, px_per_mm: float = 2.0, border_mm: float = 80.0, H: np.ndarray | None = None, shift_px: tuple[float, float] = (0.0, 0.0), guide_mm: float = 40.0, print_grid: bool = False):
    """Top-down synthetic board resting on the four square tag cards.

    Each card is a white square: the tag in the outer quadrant, diagonal to
    its board corner, and the board corner on the inner quadrant, as printed
    by tools/make_tags.py. The L lines stay visible just outside the board edges.
    With `print_grid`, dark 1.5 mm lines at `cfg.grid_lines_mm()` are printed
    on the board, like the Tomb World section lines.

    Returns (bgr image, function mm -> px) for the final image, after the
    optional homography H (3x3, acts on the flat image) and pixel shift.
    """
    w = int((cfg.width_mm + 2 * border_mm) * px_per_mm)
    h = int((cfg.height_mm + 2 * border_mm) * px_per_mm)
    img = np.full((h, w), 160, dtype=np.uint8)  # table
    o = np.array([border_mm, border_mm]) * px_per_mm  # board origin in px

    def mm2px_flat(p):
        return np.asarray(p, dtype=np.float64) * px_per_mm + o

    def fill(a_mm, b_mm, value):
        a, b = mm2px_flat(a_mm), mm2px_flat(b_mm)
        x0, x1 = sorted((int(round(a[0])), int(round(b[0]))))
        y0, y1 = sorted((int(round(a[1])), int(round(b[1]))))
        img[y0:y1, x0:x1] = value

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    card = cfg.tag_size_mm + 2 * cfg.tag_margin_mm
    for k, (tag_id, corner) in enumerate(zip(cfg.tag_ids, cfg.board_corners_mm())):
        inward = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]][k])
        ix, iy = inward
        fill(corner - inward * card, corner + inward * guide_mm, 255)
        fill(corner, corner + np.array([ix * guide_mm, -iy * 0.8]), 0)
        fill(corner, corner + np.array([-ix * 0.8, iy * guide_mm]), 0)
        tc = cfg.tag_corners_mm()[tag_id]
        t0 = mm2px_flat(tc[0]).round().astype(int)
        t2 = mm2px_flat(tc[2]).round().astype(int)
        side = t2[0] - t0[0]
        marker = cv2.aruco.generateImageMarker(dictionary, tag_id, 8 * 16)
        img[t0[1] : t0[1] + side, t0[0] : t0[0] + side] = cv2.resize(marker, (side, side), interpolation=cv2.INTER_NEAREST)
    fill((0, 0), (cfg.width_mm, cfg.height_mm), BOARD_GRAY)  # board lies on top of the cards
    if print_grid:
        xs, ys = cfg.grid_lines_mm()
        m, lw = cfg.grid_margin_mm, 0.75
        for x in xs:
            fill((x - lw, m), (x + lw, cfg.height_mm - m), LINE_GRAY)
        for y in ys:
            fill((m, y - lw), (cfg.width_mm - m, y + lw), LINE_GRAY)

    T = np.array([[1, 0, shift_px[0]], [0, 1, shift_px[1]], [0, 0, 1]], dtype=np.float64)
    M = T @ (H if H is not None else np.eye(3))
    if not np.allclose(M, np.eye(3)):
        img = cv2.warpPerspective(img, M, (w, h), borderValue=160)

    def mm2px(p):
        q = np.append(mm2px_flat(p), 1.0)
        r = M @ q
        return r[:2] / r[2]

    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), mm2px


def tomb_world_grid(**over) -> BoardConfig:
    """Test config with the Tomb World printed grid: 7 x 6 sections, 12 mm margin."""
    return make_config(width_mm=703.0, height_mm=606.0, grid_margin_mm=12.0, grid_sections=(7, 6), grid_printed=True, **over)


def mild_perspective(w: int, h: int, amount: float = 0.04) -> np.ndarray:
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * amount, h * amount * 0.5], [w * (1 - amount * 0.5), 0], [w, h], [0, h * (1 - amount)]])
    return cv2.getPerspectiveTransform(src, dst).astype(np.float64)


@pytest.fixture(autouse=True)
def no_local_camera_file(tmp_path, monkeypatch):
    """CLI tests must not read or write the machine-local data/camera.toml."""
    import battlesight.app

    monkeypatch.setattr(battlesight.app, "DEFAULT_CAMERA_FILE", tmp_path / "camera.toml")


@pytest.fixture
def cfg() -> BoardConfig:
    return make_config()
