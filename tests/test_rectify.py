import json

import cv2
import numpy as np
import pytest

from battlesight.app import main
from battlesight.camera import CameraSettings
from battlesight.calib import CharucoFinder, CharucoTarget, Lens, TagDetector, TagObs, calibrate_lens, lens_for_frame
from battlesight.pipeline import Pipeline, analyze
from battlesight.rectify import corner_angles_deg, fit_view, measure_edges, measure_grid_lines, warp_board

from .conftest import make_config, mild_perspective, render_board, tomb_world_grid

# A 4K camera 1.25 m above the board, a little off centre and tilted.
SIZE = (3840, 2160)
K = np.array([[3000.0, 0, 1920], [0, 3000.0, 1080], [0, 0, 1]])
DIST = np.array([-0.12, 0.05, 0.0005, -0.0003, 0.0])
CAMERA = np.array([330.0, 360.0, -1250.0])  # centre in board mm, z < 0 is above the table
RVEC = np.array([0.03, -0.02, 0.1])


def camera_pose():
    R, _ = cv2.Rodrigues(RVEC)
    return RVEC, -R @ CAMERA


def project(pts_mm: np.ndarray, z_mm: float, dist=None) -> np.ndarray:
    """Board points at height z_mm above the table -> image px."""
    rvec, tvec = camera_pose()
    obj = np.c_[np.asarray(pts_mm, dtype=np.float64), np.full(len(pts_mm), -z_mm)]
    px, _ = cv2.projectPoints(obj, rvec, tvec, K, dist if dist is not None else np.zeros(5))
    return px.reshape(-1, 2)


def projected_tags(cfg, dist=None) -> dict[int, TagObs]:
    out = {}
    for tag_id, quad in cfg.tag_corners_mm().items():
        c = project(quad, 0.0, dist)
        edges = np.linalg.norm(np.roll(c, -1, axis=0) - c, axis=1)
        out[tag_id] = TagObs(tag_id, c, c.mean(axis=0), float(edges.mean()))
    return out


def board_samples(cfg) -> np.ndarray:
    xs = np.linspace(0, cfg.width_mm, 7)
    ys = np.linspace(0, cfg.height_mm, 7)
    return np.array([(x, y) for x in xs for y in ys])


def lens(dist=None) -> Lens:
    return Lens(K, np.zeros(5) if dist is None else dist, SIZE, "calibrated")


# --- geometry with an exact pinhole camera -------------------------------------


def test_fit_view_exact_camera():
    cfg = make_config()
    view = fit_view(cfg, projected_tags(cfg), lens())
    assert view.reproj_max_mm < 1e-3
    assert np.allclose(view.camera_mm, [CAMERA[0], CAMERA[1], -CAMERA[2]], atol=0.5)
    pts = board_samples(cfg)
    assert np.abs(view.mm_to_px(pts) - project(pts, 0.0)).max() < 1e-3
    assert np.abs(view.px_to_mm(view.mm_to_px(pts)) - pts).max() < 1e-6


def test_thickness_correction():
    t = 3.0
    cfg = make_config(thickness_mm=t)
    obs = projected_tags(cfg)  # tags on the table
    pts = board_samples(cfg)
    surface_px = project(pts, t)  # play surface t mm above the table
    view = fit_view(cfg, obs, lens())
    assert view.surface_scale == pytest.approx(1250 / (1250 - t))
    assert np.abs(view.mm_to_px(pts) - surface_px).max() < 0.02
    assert np.abs(view.px_to_mm(surface_px) - pts).max() < 0.01
    # Without the correction a board corner is off by about r * t / h (> 1 mm).
    flat = fit_view(make_config(thickness_mm=0.0), obs, lens())
    err = np.linalg.norm(flat.px_to_mm(surface_px) - pts, axis=1)
    assert err.max() > 1.0


def test_lens_distortion_round_trip():
    l = lens(DIST)
    pts = np.array([[10.0, 20.0], [1920, 1080], [3800, 2100], [100, 2000]])
    assert np.abs(l.undistort_px(l.distort_px(pts)) - pts).max() < 1e-3


def test_fit_view_with_distortion():
    cfg = make_config(thickness_mm=2.0)
    obs = projected_tags(cfg, DIST)
    pts = board_samples(cfg)
    truth = project(pts, 2.0, DIST)
    view = fit_view(cfg, obs, lens(DIST))
    assert view.reproj_max_mm < 1e-3
    assert np.abs(view.mm_to_px(pts) - truth).max() < 0.05
    # Ignoring the distortion leaves a clear error.
    plain = fit_view(cfg, obs, lens())
    assert plain.reproj_max_mm > 0.5


def test_fit_view_needs_all_tags():
    cfg = make_config()
    obs = projected_tags(cfg)
    del obs[2]
    with pytest.raises(ValueError):
        fit_view(cfg, obs, lens())


# --- rendered boards ---------------------------------------------------------


def detect_and_fit(cfg, img, lens_=None):
    obs = TagDetector(cfg).detect(img)
    h, w = img.shape[:2]
    return fit_view(cfg, obs, lens_ or Lens.nominal((w, h), CameraSettings().focal_35mm))


def test_rectified_board_edges_and_grid():
    cfg = tomb_world_grid()
    flat, _ = render_board(cfg, print_grid=True)
    img, mm2px = render_board(cfg, H=mild_perspective(flat.shape[1], flat.shape[0]), print_grid=True)
    view = detect_and_fit(cfg, img)
    assert view.reproj_max_mm < 0.6
    pts = board_samples(cfg)
    truth = np.array([mm2px(p) for p in pts])
    px_per_mm = 2.0
    assert np.abs(view.mm_to_px(pts) - truth).max() / px_per_mm < 0.6  # mm

    rect = warp_board(img, view, cfg)
    edges = measure_edges(rect, cfg)
    for name, e in edges.items():
        assert e is not None, name
        assert abs(e.offset_mm) < 0.6, (name, e)
        assert abs(e.angle_deg) < 0.2, (name, e)
    for a in corner_angles_deg(edges):
        assert a == pytest.approx(90.0, abs=0.2)
    lines = measure_grid_lines(rect, cfg)
    assert len(lines["x"]) == 8 and len(lines["y"]) == 7
    for v in lines["x"] + lines["y"]:
        assert v is not None and abs(v) < 0.6


def test_warp_removes_lens_distortion():
    cfg = tomb_world_grid()
    flat, mm2px = render_board(cfg, print_grid=True)
    h, w = flat.shape[:2]
    f = 1.2 * w
    l = Lens(np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]]), np.array([-0.15, 0.04, 0, 0, 0]), (w, h), "calibrated")
    # Distorted image: each distorted pixel shows the flat pixel it undistorts to.
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    und = l.undistort_px(np.c_[xs.ravel(), ys.ravel()]).reshape(h, w, 2).astype(np.float32)
    img = cv2.remap(flat, und[..., 0], und[..., 1], cv2.INTER_LINEAR, borderValue=(160, 160, 160))
    rect = warp_board(img, detect_and_fit(cfg, img, l), cfg)
    edges = measure_edges(rect, cfg)
    for name, e in edges.items():
        assert e is not None and abs(e.offset_mm) < 0.7 and abs(e.angle_deg) < 0.2, (name, e)
    for v in sum(measure_grid_lines(rect, cfg).values(), []):
        assert v is not None and abs(v) < 0.7
    # Without the distortion model the straight edges come out bent: the
    # line fit gets a large residual.
    bent = measure_edges(warp_board(img, detect_and_fit(cfg, img, Lens.nominal((w, h), 24.0)), cfg), cfg)
    assert max(e.rms_mm for e in bent.values() if e is not None) > max(e.rms_mm for e in edges.values()) + 0.3


def test_measure_edges_sees_moved_board():
    """Board slid 3 mm right over still cards: the edge check shows it."""
    cfg = tomb_world_grid()
    img, mm2px = render_board(cfg)
    shift_px = int(round(3 * 2.0))
    a, b = mm2px((0, 60)), mm2px((cfg.width_mm, cfg.height_mm - 60))
    x0, x1, y0, y1 = int(a[0]), int(b[0]), int(a[1]), int(b[1])
    board = img[y0:y1, x0:x1].copy()
    img[y0:y1, x0 : x0 + shift_px] = 160
    img[y0:y1, x0 + shift_px : x1 + shift_px] = board
    edges = measure_edges(warp_board(img, detect_and_fit(cfg, img), cfg), cfg)
    assert edges["left"].offset_mm == pytest.approx(3.0, abs=0.6)
    assert edges["right"].offset_mm == pytest.approx(-3.0, abs=0.6)


def test_rectify_stage_report():
    cfg = tomb_world_grid()
    img, _ = render_board(cfg, print_grid=True)
    r = analyze(img, cfg)
    assert r.ok, r.summary()
    d = r.stages["rectify"].data
    json.dumps(r.to_dict())
    assert d["lens"]["source"] == "nominal"
    assert d["reproj_max_mm"] < 0.6
    assert d["rectified"]["size"] == [int(round((703 + 100) * 2)), int(round((606 + 100) * 2))]
    assert all(abs(a - 90) < 0.2 for a in d["corner_angles_deg"])
    assert len(d["grid_lines"]["x"]) == 8
    assert np.array(d["H_mm_to_px"]).shape == (3, 3)
    views = Pipeline(cfg).views(img, r)
    assert views["rectified"].shape[:2] == (1412, 1606)


def test_rectify_stage_no_grid_check_without_printed_grid():
    cfg = make_config()
    img, _ = render_board(cfg)
    r = analyze(img, cfg)
    assert r.ok and r.stages["rectify"].data["grid_lines"] is None


def test_rectify_stage_hints_swapped_size():
    cfg = tomb_world_grid()
    img, _ = render_board(cfg)
    wrong = make_config(width_mm=606.0, height_mm=703.0)
    r = analyze(img, wrong)
    assert not r.stages["rectify"].ok
    assert "swapped" in r.stages["rectify"].message


def test_rectify_stage_fails_on_bad_fit():
    # Tags printed at 36 mm while the config says 40 mm: the corners do not fit one plane.
    img, _ = render_board(make_config(tag_size_mm=36.0, tag_margin_mm=7.0))
    r = analyze(img, make_config(max_reproj_mm=0.5))
    assert r.stages["corners"].ok
    assert not r.stages["rectify"].ok
    assert "fit rms" in r.stages["rectify"].message


# --- lens files and calibration --------------------------------------------


def test_lens_for_frame(tmp_path):
    nominal = lens_for_frame(CameraSettings(), (4000, 3000))
    assert nominal.source == "nominal" and not nominal.has_distortion
    assert nominal.K[0, 0] == pytest.approx(24 / 43.267 * 5000)
    assert lens_for_frame(CameraSettings(focal_35mm=26), (4000, 3000)).K[0, 0] == pytest.approx(26 / 43.267 * 5000)

    path = tmp_path / "camera.toml"
    CameraSettings(source=1).with_lens(Lens(K, DIST, SIZE, "calibrated", 0.3), views=12).save(path)
    camera = CameraSettings.load(path)
    assert camera.source == 1 and camera.lens_views == 12 and camera.lens_date
    same = lens_for_frame(camera, SIZE)
    assert same.source == "calibrated" and np.allclose(same.dist, DIST) and same.rms_px == 0.3
    half = lens_for_frame(camera, (1920, 1080))
    assert half.source == "calibrated" and half.K[0, 0] == pytest.approx(1500) and half.K[1, 2] == pytest.approx(540)
    assert lens_for_frame(camera, (3072, 4096)).source == "nominal"  # other aspect ratio
    assert lens_for_frame(CameraSettings.load(tmp_path / "missing.toml"), SIZE).source == "nominal"


def charuco_views(n: int = 16, seed: int = 1):
    """Images of the ChArUco target from an ideal camera (f = 1100 px) at random poses."""
    target = CharucoTarget()
    ppm = 4.0
    bw, bh = (int(round(s * target.square_mm * ppm)) for s in target.squares)
    board = target.board().generateImage((bw, bh), marginSize=0, borderBits=1)
    page = cv2.copyMakeBorder(board, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    size = (1280, 960)
    Kc = np.array([[1100.0, 0, 640], [0, 1100.0, 480], [0, 0, 1]])
    rng = np.random.default_rng(seed)
    centre = np.array([bw / ppm / 2, bh / ppm / 2, 0])
    page_to_mm = np.array([[1 / ppm, 0, -40 / ppm], [0, 1 / ppm, -40 / ppm], [0, 0, 1]])
    images = []
    for _ in range(n):
        rvec = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(-0.3, 0.3)])
        R, _ = cv2.Rodrigues(rvec)
        t = -R @ centre + np.array([rng.uniform(-150, 150), rng.uniform(-110, 110), rng.uniform(480, 620)])
        Hc = Kc @ np.c_[R[:, 0], R[:, 1], t] @ page_to_mm
        img = cv2.warpPerspective(page, Hc, size, flags=cv2.INTER_AREA, borderValue=128)
        images.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    return images, size, Kc


def test_calibrate_lens_synthetic():
    images, size, Kc = charuco_views()
    finder = CharucoFinder()
    views = [v for v in (finder.detect(img) for img in images) if v is not None]
    assert len(views) >= 10
    l = calibrate_lens(views, size)
    assert l.rms_px < 0.5
    assert l.K[0, 0] == pytest.approx(1100, rel=0.02)
    assert l.K[1, 1] == pytest.approx(1100, rel=0.02)
    assert abs(l.K[0, 2] - 640) < 15 and abs(l.K[1, 2] - 480) < 15
    # An ideal camera: distortion stays small over the image.
    pts = np.array([[20.0, 20.0], [1260, 940], [640, 480]])
    assert np.abs(l.distort_px(pts) - pts).max() < 3.0


def test_charuco_not_taken_for_corner_tags():
    images, _, _ = charuco_views(n=1)
    assert TagDetector(make_config()).detect(images[0]) == {}


def test_cli_calibrate_images(tmp_path, capsys):
    images, size, _ = charuco_views()
    for i, img in enumerate(images):
        cv2.imwrite(str(tmp_path / f"v{i:02d}.png"), img)
    out = tmp_path / "camera.toml"
    CameraSettings(source="http://phone:8080/video", focal_35mm=26.0).save(out)
    assert main(["calibrate", "--images", str(tmp_path), "--camera-file", str(out)]) == 0
    camera = CameraSettings.load(out)
    assert camera.lens.size == size and camera.lens_views >= 10
    assert camera.lens.K[0, 0] == pytest.approx(1100, rel=0.02)
    # Only [lens] changed: the saved source and focal length stay.
    assert camera.source == "http://phone:8080/video" and camera.focal_35mm == 26.0
    # Too few views: exit 1, no file.
    few = tmp_path / "few"
    few.mkdir()
    cv2.imwrite(str(few / "a.png"), images[0])
    assert main(["calibrate", "--images", str(few), "--camera-file", str(tmp_path / "x.toml")]) == 1
    assert not (tmp_path / "x.toml").exists()
    assert main(["calibrate", "--images", str(tmp_path / "nothing")]) == 2
