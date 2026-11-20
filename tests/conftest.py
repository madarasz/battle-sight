"""Synthetic board images for tests. No camera needed."""

from __future__ import annotations

import copy
import hashlib
import os
from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from battlesight.config import BoardConfig

BENCH_TABLE = pytest.StashKey[str]()  # bases benchmark table of the photo tests
IDENT_TABLE = pytest.StashKey[str]()  # bases benchmark table of the photo tests

BOARD_GRAY = 70  # dark board, like Tomb World
LINE_GRAY = 30  # printed grid lines on the board
RIM_GRAY = 15  # black base rim
BASE_TOP_GRAY = 140  # painted base top (scenic)
MODEL_GRAY = 190  # model body over the base
WALL_GRAY = 25  # wall pieces


@dataclass(frozen=True)
class Base:
    """A round base in the synthetic image. `model` covers the rim from
    `model_deg` for `model_span_deg` degrees (0 deg = +x, 90 deg = +y)."""

    center_mm: tuple[float, float]
    diameter_mm: float
    rim_mm: float = 2.0  # visible black bevel
    model_deg: float = 0.0
    model_span_deg: float = 0.0
    rim_gray: int = RIM_GRAY


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


def render_board(
    cfg: BoardConfig,
    px_per_mm: float = 2.0,
    border_mm: float = 80.0,
    H: np.ndarray | None = None,
    shift_px: tuple[float, float] = (0.0, 0.0),
    guide_mm: float = 40.0,
    print_grid: bool = False,
    bases: tuple[Base, ...] = (),
    walls: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = (),
    discs: tuple[tuple[tuple[float, float], float, int], ...] = (),
):
    """Top-down synthetic board resting on the four square tag cards.

    Each card is a white square: the tag in the outer quadrant, diagonal to
    its board corner, and the board corner on the inner quadrant, as printed
    by tools/make_tags.py. The L lines stay visible just outside the board edges.
    With `print_grid`, dark 1.5 mm lines at `cfg.grid_lines_mm()` are printed
    on the board, like the Tomb World section lines. `discs` are flat round
    tokens or prints ((x, y) mm, diameter mm, grey). `walls` are dark
    rectangles (corner mm, opposite corner mm). `bases` are drawn last, with
    their black rim, a lighter top, and an optional model over part of the rim.

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
    sub = 16  # fixed-point shift for anti-aliased circles

    def disc(center_mm, r_mm, value):
        c = np.rint(mm2px_flat(center_mm) * sub).astype(int)
        cv2.circle(img, (int(c[0]), int(c[1])), int(round(r_mm * px_per_mm * sub)), int(value), -1, cv2.LINE_AA, 4)

    for center, d, value in discs:
        disc(center, 0.5 * d, value)
    for a, b in walls:
        fill(a, b, WALL_GRAY)
    for base in bases:
        r = 0.5 * base.diameter_mm
        disc(base.center_mm, r, base.rim_gray)
        disc(base.center_mm, r - base.rim_mm, BASE_TOP_GRAY)
        if base.model_span_deg > 0:
            c = mm2px_flat(base.center_mm)
            ax = int(round((r + 3.0) * px_per_mm))
            a0 = base.model_deg - 0.5 * base.model_span_deg
            cv2.ellipse(img, (int(round(c[0])), int(round(c[1]))), (ax, ax), 0, a0, a0 + base.model_span_deg, MODEL_GRAY, -1, cv2.LINE_AA)

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


def _digest(a: np.ndarray | None):
    if a is None:
        return None
    a = np.ascontiguousarray(a)
    return a.shape, a.dtype.str, hashlib.blake2b(a.tobytes(), digest_size=16).hexdigest()


@pytest.fixture(autouse=True, scope="session")
def find_circles_memo():
    """Keeps the result of each `circles.find_circles` call for the test
    process. Many tests run the same synthetic board through the pipeline;
    the function is deterministic, so a repeated call gives a copy of the
    first result. The key is the image, the parameters, the region and the
    foreground. Set BATTLESIGHT_TEST_NO_MEMO=1 to compute each call."""
    if os.environ.get("BATTLESIGHT_TEST_NO_MEMO"):
        yield
        return
    import battlesight.circles as circles

    compute = circles.find_circles
    memo: dict = {}

    def find_circles(img, p, region=None, foreground=None):
        key = (_digest(img), p, _digest(region), _digest(foreground))
        if key not in memo:
            memo[key] = compute(img, p, region, foreground)
        return copy.deepcopy(memo[key])

    circles.find_circles = find_circles
    yield
    circles.find_circles = compute


@pytest.fixture(autouse=True)
def no_local_camera_file(tmp_path, monkeypatch):
    """CLI tests must not read or write the machine-local data/camera.toml."""
    import battlesight.app

    monkeypatch.setattr(battlesight.app, "DEFAULT_CAMERA_FILE", tmp_path / "camera.toml")


@pytest.fixture(autouse=True)
def no_local_gallery(request, tmp_path, monkeypatch):
    """Unit tests must not load the real data/teams (and the DINOv2 model).
    The photo tests keep it: their ident benchmark needs it."""
    if request.node.get_closest_marker("photos") is not None:
        return
    import battlesight.ident

    monkeypatch.setattr(battlesight.ident, "DEFAULT_GALLERY_ROOT", tmp_path / "gallery")


@pytest.fixture
def cfg() -> BoardConfig:
    return make_config()


# Test category of each test file (CLAUDE.md "Which tests to run"). A new
# test file must be added here; the photo tests have their own marker.
CATEGORIES = {
    "core": ("test_calib", "test_camera", "test_colors", "test_config", "test_measure", "test_moves", "test_score", "test_scrcpy", "test_shoot", "test_terrain", "test_walls"),
    "vision": ("test_bases", "test_bench", "test_ident", "test_ident_bench", "test_lightfit", "test_pipeline", "test_rectify"),
    "live": ("test_live",),
    "ui": ("test_ui",),
}
_CATEGORY_OF = {name: cat for cat, names in CATEGORIES.items() for name in names}


@pytest.hookimpl(tryfirst=True)  # before `-m` selects the tests
def pytest_collection_modifyitems(config, items):
    for item in items:
        name = item.path.stem
        if name == "test_photos":
            continue
        if name not in _CATEGORY_OF:
            raise pytest.UsageError(f"tests/{item.path.name} has no category: add it to CATEGORIES in tests/conftest.py")
        item.add_marker(_CATEGORY_OF[name])


_TABLES: dict[str, str] = {}  # benchmark tables, in the main process (also with xdist)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Sends a benchmark table from the process that made it (an xdist
    worker or the main process) to the main process, in the report."""
    outcome = yield
    if call.when != "teardown":
        return
    report = outcome.get_result()
    for key, name in ((BENCH_TABLE, "bench_table"), (IDENT_TABLE, "ident_table")):
        table = item.config.stash.get(key, None)
        if table:
            report.user_properties.append((name, table))
            del item.config.stash[key]


def pytest_runtest_logreport(report):
    for name, value in report.user_properties:
        if name in ("bench_table", "ident_table"):
            _TABLES[name] = value


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print the benchmark tables after the photo tests. Nothing when they
    did not run."""
    table = _TABLES.get("bench_table")
    if table:
        terminalreporter.write_sep("=", "bases benchmark (docs/specs/bases-benchmark.md)")
        terminalreporter.write_line(table)
    table = _TABLES.get("ident_table")
    if table:
        terminalreporter.write_sep("=", "identification benchmark")
        terminalreporter.write_line(table)
