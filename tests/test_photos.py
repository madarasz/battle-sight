"""Regression tests on real photos in tests/images.

Each photo <name>.jpg has a reviewed report <name>.expected.json next to it
(the <name>.json that `battlesight image --out` writes). The tests run the
pipeline on the photo and compare the result with the expected report, within
a tolerance in board millimetres.

Optional files next to a photo:

- <name>.bases.json: hand-checked base positions and sizes (ground truth).
- <name>.empty.jpg: the empty board (terrain, no operatives or tokens) from
  the same camera and light. The bases stage then runs in reference mode.

`bases_baseline.json` holds the bases benchmark metrics (tools/bench_bases.py
--write-baseline). The metrics table prints after the photo tests.

Marked `photos`. The default `uv run pytest` skips them. Run them with:

    uv run pytest -m photos
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from battlesight import bench
from battlesight.bench import REFERENCE_TAG, reference_path
from battlesight.config import DEFAULT_CONFIG, BoardConfig
from battlesight.pipeline import FrameResult, Pipeline

from .conftest import BENCH_TABLE

pytestmark = pytest.mark.photos

SHOTS = Path(__file__).resolve().parents[1] / "tests" / "images"
PHOTOS = bench.find_photos([SHOTS]) if SHOTS.is_dir() else []
BASELINE = SHOTS / "bases_baseline.json"

CORNER_TOL_MM = 1.0  # max distance between found and expected corner points


def expected_path(photo: Path) -> Path:
    return photo.with_name(f"{photo.stem}.expected.json")


@pytest.fixture(scope="module")
def cfg() -> BoardConfig:
    return BoardConfig.load(DEFAULT_CONFIG)


@pytest.fixture(scope="module")
def results(cfg) -> dict[Path, FrameResult]:
    """Pipeline results, computed once per photo for all tests in this module."""

    class Lazy(dict):
        def __missing__(self, photo: Path) -> FrameResult:
            frame = cv2.imread(str(photo), cv2.IMREAD_COLOR)
            assert frame is not None, f"cannot read {photo}"
            ref = reference_path(photo)
            pipe = Pipeline(cfg, reference=None if ref is None else cv2.imread(str(ref), cv2.IMREAD_COLOR))
            self[photo] = pipe.analyze(frame, source=str(photo))
            return self[photo]

    return Lazy()


def load_expected(photo: Path) -> dict:
    path = expected_path(photo)
    if not path.is_file():
        pytest.fail(
            f"no {path.name}. Run `uv run battlesight image {photo} --out out/`, check the overlay, "
            f"then copy out/{photo.stem}.json to {path}"
        )
    return json.loads(path.read_text())


def _px_per_mm(cfg: BoardConfig, tags: dict, tag_id: int) -> float:
    return tags[str(tag_id)]["side_px"] / cfg.tag_size_mm


photo_params = pytest.mark.parametrize("photo", PHOTOS, ids=lambda p: p.name)
needs_photos = pytest.mark.skipif(not PHOTOS, reason="no photos in tests/images")


@needs_photos
@photo_params
def test_photo_passes_all_stages(results, photo):
    r = results[photo]
    assert r.ok, r.summary()


@needs_photos
@photo_params
def test_photo_size_matches(results, photo):
    exp = load_expected(photo)
    r = results[photo]
    assert [r.width, r.height] == exp["size"], "image size or EXIF rotation changed"


@needs_photos
@photo_params
def test_corners_match_expected(cfg, results, photo):
    exp = load_expected(photo)["stages"]["corners"]["data"]
    got = results[photo].stages["corners"].data
    assert sorted(got["tags"]) == sorted(exp["tags"]), "found tag ids differ"

    errors = []
    for k, tag_id in enumerate(cfg.tag_ids):
        e, g = exp["board_corners_px"][k], got["board_corners_px"][k]
        if e is None or g is None:
            assert e == g, f"board corner {k}: expected {e}, got {g}"
            continue
        mm = float(np.linalg.norm(np.subtract(g, e))) / _px_per_mm(cfg, exp["tags"], tag_id)
        if mm > CORNER_TOL_MM:
            errors.append(f"board corner {k} off by {mm:.2f} mm")
    for tid, et in exp["tags"].items():
        d = np.linalg.norm(np.subtract(got["tags"][tid]["corners_px"], et["corners_px"]), axis=1)
        mm = float(d.max()) / _px_per_mm(cfg, exp["tags"], int(tid))
        if mm > CORNER_TOL_MM:
            errors.append(f"tag {tid} corner off by {mm:.2f} mm")
    assert not errors, "; ".join(errors) + f" (tolerance {CORNER_TOL_MM} mm)"


# --- rectify (priority 2) ---------------------------------------------------

RECTIFY_TOL_MM = 1.0  # max board-plane distance between found and expected homography, edge and grid line
EDGE_ANGLE_TOL_DEG = 0.3  # board edges straight and aligned with the board axes
CORNER_ANGLE_TOL_DEG = 0.5  # board corners at 90 deg
SPACING_TOL_MM = 1.5  # spacing of neighbouring printed grid lines (Tomb World: 97 mm, 0.4 mm per inch)
GRID_TOL_MM = 3.5  # printed grid line from its nominal place; includes card placement error


def _board_points(cfg: BoardConfig) -> np.ndarray:
    xs = np.linspace(0, cfg.width_mm, 9)
    ys = np.linspace(0, cfg.height_mm, 9)
    return np.array([(x, y) for x in xs for y in ys])


@needs_photos
@photo_params
def test_rectify_match_expected(cfg, results, photo):
    exp = load_expected(photo)["stages"]["rectify"]["data"]
    got = results[photo].stages["rectify"].data
    assert got["lens"]["source"] == exp["lens"]["source"], "lens changed: calibration file added or removed?"

    # Board points through the expected homography, back through the found one.
    pts = _board_points(cfg).reshape(-1, 1, 2)
    px = cv2.perspectiveTransform(pts, np.array(exp["H_mm_to_px"]))
    back = cv2.perspectiveTransform(px, np.linalg.inv(np.array(got["H_mm_to_px"])))
    mm = float(np.linalg.norm(back - pts, axis=2).max())
    assert mm <= RECTIFY_TOL_MM, f"board plane moved {mm:.2f} mm (tolerance {RECTIFY_TOL_MM} mm)"

    errors = []
    for name, e in exp["edges"].items():
        g = got["edges"][name]
        if e is None or g is None:
            if (e is None) != (g is None):
                errors.append(f"edge {name}: expected {e}, got {g}")
            continue
        if abs(g["offset_mm"] - e["offset_mm"]) > RECTIFY_TOL_MM:
            errors.append(f"edge {name} offset {g['offset_mm']:.2f} mm, expected {e['offset_mm']:.2f} mm")
    if exp["grid_lines"] is not None:
        for key in ("x", "y"):
            for k, (e, g) in enumerate(zip(exp["grid_lines"][key], got["grid_lines"][key])):
                if e is not None and g is not None and abs(g - e) > RECTIFY_TOL_MM:
                    errors.append(f"grid line {key}{k} offset {g:.2f} mm, expected {e:.2f} mm")
                elif (e is None) != (g is None):
                    errors.append(f"grid line {key}{k}: expected {e}, got {g}")
    assert not errors, "; ".join(errors) + f" (tolerance {RECTIFY_TOL_MM} mm)"


@needs_photos
@photo_params
def test_rectify_done_criteria(cfg, results, photo):
    """Plan priority 2, done when: board edges straight, corners at 90 deg,
    grid lines on the printed grid, known distances right anywhere."""
    d = results[photo].stages["rectify"].data
    errors = []
    for name, e in d["edges"].items():
        if e is None:
            errors.append(f"edge {name} not found")
        elif abs(e["angle_deg"]) > EDGE_ANGLE_TOL_DEG:
            errors.append(f"edge {name} tilted {e['angle_deg']:.2f} deg")
    for k, a in enumerate(d["corner_angles_deg"]):
        if a is None or abs(a - 90.0) > CORNER_ANGLE_TOL_DEG:
            errors.append(f"corner {k} angle {a}")
    if d["grid_lines"] is not None:
        from battlesight.rectify import grid_spacing_errors_mm

        for key in ("x", "y"):
            for k, v in enumerate(d["grid_lines"][key]):
                if v is None:
                    errors.append(f"grid line {key}{k} not found")
                elif abs(v) > GRID_TOL_MM:
                    errors.append(f"grid line {key}{k} off by {v:.2f} mm")
        worst = max(map(abs, grid_spacing_errors_mm(cfg, d["grid_lines"])), default=0.0)
        if worst > SPACING_TOL_MM:
            errors.append(f"grid spacing off by {worst:.2f} mm")
    assert not errors, "; ".join(errors)


# --- bases (priority 3) -----------------------------------------------------

BASES_TOL_MM = 1.0  # found base centre from its expected centre (regression)


def load_truth(photo: Path) -> bench.GroundTruth:
    path = bench.truth_path(photo)
    if not path.is_file():
        pytest.skip(f"no {path.name}: no base ground truth for this photo")
    return bench.load_truth(path)


def _dist(a, b) -> float:
    return float(np.linalg.norm(np.subtract(a, b)))


def photo_score(photo: Path, r: FrameResult, gt: bench.GroundTruth) -> bench.PhotoScore:
    st = r.stages["bases"]
    return bench.score(photo.stem, gt, st.data["bases"], st.data["mode"], st.data["changed_fraction"], st.ms)


@pytest.fixture(scope="module")
def bench_scores(request, results) -> list[bench.PhotoScore]:
    """Benchmark scores of all photos with ground truth, from the `results`
    fixture (no second pipeline run). Stores the metrics table for the
    terminal summary (tests/conftest.py)."""
    scores = [photo_score(p, results[p], bench.load_truth(bench.truth_path(p))) for p in PHOTOS if bench.truth_path(p).is_file()]
    if scores:
        table = bench.format_table(scores)
        details = bench.format_details(scores)
        request.config.stash[BENCH_TABLE] = table + ("\n\n" + details if details else "")
    return scores


@needs_photos
@photo_params
def test_bases_match_expected(results, photo):
    exp = load_expected(photo)["stages"]["bases"]["data"]
    got = results[photo].stages["bases"].data
    assert got["mode"] == exp["mode"], "empty-board reference added or removed?"
    errors = []
    left = list(got["bases"])
    for e in exp["bases"]:
        near = [g for g in left if _dist(g["center_mm"], e["center_mm"]) <= BASES_TOL_MM]
        if not near:
            errors.append(f"expected {e['diameter_mm']:g} mm base at {e['center_mm']} not found")
            continue
        g = min(near, key=lambda g: _dist(g["center_mm"], e["center_mm"]))
        left.remove(g)
        if g["diameter_mm"] != e["diameter_mm"]:
            errors.append(f"base at {e['center_mm']}: {g['diameter_mm']:g} mm, expected {e['diameter_mm']:g} mm")
    errors += [f"new {g['diameter_mm']:g} mm base at {g['center_mm']}" for g in left]
    assert not errors, "; ".join(errors) + f" (tolerance {BASES_TOL_MM} mm)"


@needs_photos
@photo_params
def test_bases_clear_rims_found(results, photo):
    """Plan priority 3 for the bases with a clearly visible rim, in any mode:
    found within `tol_mm`, right size."""
    clear = [t for t in load_truth(photo).bases if t.rim == "clear"]
    matches, missed, _ = bench.match(clear, results[photo].stages["bases"].data["bases"])
    errors = [f"{t.note}: not found within {t.tol_mm:g} mm" for t in missed]
    errors += [f"{m.truth.note}: {m.found['diameter_mm']:g} mm, true {m.truth.diameter_mm:g} mm" for m in matches if m.size_ok is False]
    assert not errors, "; ".join(errors)


@needs_photos
def test_bases_benchmark_no_worse(cfg, bench_scores):
    """Benchmark totals not worse than tests/images/bases_baseline.json
    (rules in specs/bases-benchmark.md section 7.2)."""
    if not bench_scores:
        pytest.skip("no photo with <name>.bases.json")
    if not BASELINE.is_file():
        pytest.skip(f"no {BASELINE.name}: run `uv run tools/bench_bases.py --write-baseline {BASELINE.relative_to(SHOTS.parents[2])}`")
    cmp = bench.compare(bench.report(bench_scores, bench.bases_values(cfg)), json.loads(BASELINE.read_text()))
    assert cmp.ok, "; ".join(cmp.worse)


@needs_photos
@photo_params
def test_bases_done_criteria(results, photo):
    """Plan priority 3, done when: all bases found on a board with walls,
    centre error under 2 mm, correct diameter class, no false bases. Needs
    the empty-board reference <name>.empty.jpg. True bases with a hidden rim
    count with their own `tol_mm` (their true position is known less well)."""
    if reference_path(photo) is None:
        pytest.skip(f"no {photo.stem}{REFERENCE_TAG}.jpg: the done test needs an empty-board reference")
    s = photo_score(photo, results[photo], load_truth(photo))
    assert s.done_ok, bench.format_details([s])


# --- priority 4: operative identity -------------------------------------------


@needs_photos
@photo_params
def test_ident_done_criteria(cfg, photo):
    """Plan priority 4, done when: 9 of 10 bases named right on the first try.
    Needs <name>.ident.json (teams and the true name at each base centre) and
    the gallery of those teams in data/teams/. Runs the full pipeline with
    the gallery (and the empty-board reference when there is one)."""
    truth_path = photo.with_name(f"{photo.stem}.ident.json")
    if not truth_path.is_file():
        pytest.skip(f"no {truth_path.name}: the done test needs the true operative names")
    from battlesight import ident as idt

    truth = idt.IdentTruth.load(truth_path)
    missing = [t for t in truth.teams if not (idt.DEFAULT_GALLERY_ROOT / t).is_dir()]
    if missing:
        pytest.skip(f"no gallery for {missing} in {idt.DEFAULT_GALLERY_ROOT}")
    gallery = idt.Gallery.for_teams(truth.teams, cfg)
    frame = cv2.imread(str(photo), cv2.IMREAD_COLOR)
    ref = reference_path(photo)
    pipe = Pipeline(cfg, reference=None if ref is None else cv2.imread(str(ref), cv2.IMREAD_COLOR), gallery=gallery)
    r = pipe.analyze(frame, source=str(photo))
    assert r.ok, r.summary()
    centers = np.array([b.center_mm for b in r.scratch["bases"]]).reshape(-1, 2)
    s = idt.score_truth(truth, centers, r.scratch.get("idents", []))
    assert s["done_ok"], s
