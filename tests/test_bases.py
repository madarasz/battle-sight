"""Priority 3: bases. Circle fit of known diameters, with and without an
empty-board reference. Synthetic boards from tests/conftest.py."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from battlesight import circles
from battlesight.app import main
from battlesight.config import BoardConfig
from battlesight.pipeline import BASES_MARGIN_MM, Pipeline, analyze, bases_image

from .conftest import Base, make_config, mild_perspective, render_board

CENTER_TOL_MM = 0.5  # synthetic images: exact rims, so much tighter than the 2 mm done test


def top_down(cfg: BoardConfig, **kw) -> np.ndarray:
    """Synthetic board drawn straight at the bases scale and border, so
    find_circles can run on it without the corners and rectify stages."""
    p, m, border = cfg.bases_px_per_mm, BASES_MARGIN_MM, 80.0
    img, _ = render_board(cfg, px_per_mm=p, border_mm=border, **kw)
    k = int(round((border - m) * p))  # crop the table border down to the bases margin
    w, h = (int(round((v + 2 * m) * p)) for v in (cfg.width_mm, cfg.height_mm))
    return img[k : k + h, k : k + w]


def params(cfg: BoardConfig) -> circles.CircleParams:
    return circles.CircleParams.from_config(cfg, BASES_MARGIN_MM)


def match(found, base: Base, tol: float = CENTER_TOL_MM):
    """The found circle nearest to `base`, when it is within tol mm."""
    if not found:
        return None
    c = min(found, key=lambda c: np.linalg.norm(c.center_mm - base.center_mm))
    return c if np.linalg.norm(c.center_mm - base.center_mm) <= tol else None


@pytest.fixture
def cfg() -> BoardConfig:
    return make_config(width_mm=400.0, height_mm=300.0)


@pytest.mark.parametrize("d", [25.0, 28.0, 32.0, 40.0, 50.0])
def test_each_size_found_with_center_and_class(cfg, d):
    base = Base((150.3, 120.7), d)
    found = circles.find_circles(top_down(cfg, bases=(base,)), params(cfg))
    assert len(found) == 1
    c = found[0]
    assert np.linalg.norm(c.center_mm - base.center_mm) < CENTER_TOL_MM
    assert c.diameter_mm == d
    assert abs(c.fit_diameter_mm - d) < 1.0
    assert c.coverage > 0.9


def test_neighbour_sizes_told_apart(cfg):
    bases = (Base((80, 80), 25), Base((150, 80), 28), Base((220, 80), 32))
    found = circles.find_circles(top_down(cfg, bases=bases), params(cfg))
    assert len(found) == 3
    for b in bases:
        c = match(found, b)
        assert c is not None and c.diameter_mm == b.diameter_mm


def test_model_over_part_of_the_rim(cfg):
    """A model hides 40 % of the rim (144 deg). The visible rim is enough."""
    base = Base((200, 150), 32, model_deg=30, model_span_deg=144)
    found = circles.find_circles(top_down(cfg, bases=(base,)), params(cfg))
    c = match(found, base)
    assert c is not None and c.diameter_mm == 32
    assert 0.5 <= c.coverage < 0.7


def test_rim_mostly_hidden_not_reported(cfg):
    """Less visible rim than `bases.min_coverage`: no guess."""
    base = Base((200, 150), 32, model_deg=0, model_span_deg=220)
    assert circles.find_circles(top_down(cfg, bases=(base,)), params(cfg)) == []


def test_touching_bases_both_found(cfg):
    bases = (Base((150, 150), 40), Base((190, 150), 40), Base((170, 184.7), 40))
    found = circles.find_circles(top_down(cfg, bases=bases), params(cfg))
    assert len(found) == 3
    assert all(match(found, b) is not None for b in bases)


def test_base_against_a_wall(cfg):
    """A wall touches the base. The wall is no base, the base is still found."""
    base = Base((200, 150), 40)
    walls = (((222.0, 60.0), (230.0, 240.0)), ((40.0, 40.0), (360.0, 46.0)))
    found = circles.find_circles(top_down(cfg, bases=(base,), walls=walls), params(cfg))
    assert len(found) == 1 and match(found, base) is not None


def test_light_tokens_and_other_sizes_rejected(cfg):
    """A light 40 mm token (no dark rim) and a dark 60 mm disc (no base size)."""
    discs = (((100.0, 100.0), 40.0, 200), ((260.0, 160.0), 60.0, 15))
    assert circles.find_circles(top_down(cfg, discs=discs), params(cfg)) == []


def test_printed_grid_gives_no_bases(cfg):
    from .conftest import tomb_world_grid

    g = tomb_world_grid()
    assert circles.find_circles(top_down(g, print_grid=True), params(g)) == []


def test_reference_removes_static_circles(cfg):
    """A dark 40 mm print on the board looks like a base in one frame. The
    empty-board reference has it too, so reference mode ignores it."""
    static = (((100.0, 100.0), 40.0, 15),)
    walls = (((40.0, 200.0), (360.0, 206.0)),)
    base = Base((250, 120), 32, model_deg=200, model_span_deg=90)
    ref = top_down(cfg, discs=static, walls=walls)
    img = top_down(cfg, discs=static, walls=walls, bases=(base,))

    single = circles.find_circles(img, params(cfg))
    assert len(single) == 2

    fg = circles.foreground_mask(img, ref, cfg.bases_diff_threshold)
    found = circles.find_circles(img, params(cfg), foreground=fg)
    assert len(found) == 1 and match(found, base) is not None


def test_foreground_mask_ignores_noise(cfg):
    ref = top_down(cfg)
    rng = np.random.default_rng(1)
    noisy = np.clip(ref.astype(np.int16) + rng.normal(0, 4, ref.shape), 0, 255).astype(np.uint8)
    assert circles.foreground_mask(noisy, ref).mean() < 0.001


def test_region_limits_centres(cfg):
    base = Base((200, 150), 32)
    img = top_down(cfg, bases=(base,))
    region = np.zeros(img.shape[:2], bool)
    region[:, : img.shape[1] // 4] = True  # left quarter only, base not in it
    assert circles.find_circles(img, params(cfg), region=region) == []


# --- pipeline stage -------------------------------------------------------------


def camera_view(cfg: BoardConfig, **kw):
    """Board in a camera-like view: 2 px per mm, mild perspective."""
    img, mm2px = render_board(cfg, **kw)
    h, w = img.shape[:2]
    Hp = mild_perspective(w, h)
    img, mm2px = render_board(cfg, H=Hp, **kw)
    return img, mm2px


def test_bases_stage_single_frame(cfg):
    bases = (Base((120.0, 100.0), 32, model_deg=90, model_span_deg=100), Base((280.0, 200.0), 40))
    img, mm2px = camera_view(cfg, bases=bases)
    r = analyze(img, cfg)
    assert r.ok, r.summary()
    d = r.stages["bases"].data
    assert d["mode"] == "single_frame" and d["changed_fraction"] is None
    assert len(d["bases"]) == 2
    for b in bases:
        got = min(d["bases"], key=lambda g: np.linalg.norm(np.subtract(g["center_mm"], b.center_mm)))
        assert np.linalg.norm(np.subtract(got["center_mm"], b.center_mm)) < 1.0
        assert got["diameter_mm"] == b.diameter_mm
        assert np.linalg.norm(np.subtract(got["center_px"], mm2px(b.center_mm))) < 2.0
    assert "2 bases" in r.stages["bases"].message
    json.dumps(r.to_dict())


def test_bases_stage_with_reference(cfg):
    static = (((100.0, 220.0), 40.0, 15),)
    base = Base((250.0, 120.0), 28)
    empty, _ = camera_view(cfg, discs=static)
    img, _ = camera_view(cfg, discs=static, bases=(base,))
    pipe = Pipeline(cfg, reference=empty)
    assert pipe.has_reference
    r = pipe.analyze(img)
    d = r.stages["bases"].data
    assert d["mode"] == "reference" and 0 < d["changed_fraction"] < 0.05
    assert [b["diameter_mm"] for b in d["bases"]] == [28.0]
    pipe.clear_reference()
    assert len(pipe.analyze(img).stages["bases"].data["bases"]) == 2


def test_bad_reference_rejected(cfg):
    with pytest.raises(ValueError, match="reference"):
        Pipeline(cfg, reference=np.full((300, 300, 3), 160, np.uint8))


def test_bases_stage_no_bases_is_ok(cfg):
    img, _ = camera_view(cfg)
    r = analyze(img, cfg)
    assert r.ok and r.stages["bases"].data["bases"] == []


def test_bases_image_scale(cfg):
    img, _ = render_board(cfg)
    r = analyze(img, cfg, until="rectify")
    b = bases_image(img, r.scratch["view"], cfg)
    p = cfg.bases_px_per_mm
    assert b.shape[:2] == (round((cfg.height_mm + 2 * BASES_MARGIN_MM) * p), round((cfg.width_mm + 2 * BASES_MARGIN_MM) * p))


def test_cli_image_reference(tmp_path, capsys):
    toml = tmp_path / "board.toml"
    toml.write_text('[board]\nwidth_mm = 400.0\nheight_mm = 300.0\n[tags]\nids = [0, 1, 2, 3]\nsize_mm = 40.0\nmargin_mm = 5.0\n')
    cfg = BoardConfig.load(toml)
    static = (((100.0, 220.0), 40.0, 15),)
    empty, _ = camera_view(cfg, discs=static)
    img, _ = camera_view(cfg, discs=static, bases=(Base((250.0, 120.0), 28),))
    cv2.imwrite(str(tmp_path / "empty.png"), empty)
    cv2.imwrite(str(tmp_path / "game.png"), img)
    out = tmp_path / "out"
    args = ["image", str(tmp_path / "game.png"), "--config", str(toml), "--json", "-", "-q", "--out", str(out)]
    assert main(args + ["--reference", str(tmp_path / "empty.png")]) == 0
    report = json.loads(capsys.readouterr().out)
    d = report["images"][0]["stages"]["bases"]["data"]
    assert d["mode"] == "reference" and len(d["bases"]) == 1
    assert report["reference"] == str(tmp_path / "empty.png")
    assert (out / "game.bases.jpg").is_file()

    assert main(args) == 0  # no reference: the static print is found too
    assert len(json.loads(capsys.readouterr().out)["images"][0]["stages"]["bases"]["data"]["bases"]) == 2

    assert main(args + ["--reference", str(tmp_path / "missing.png")]) == 2
    cv2.imwrite(str(tmp_path / "blank.png"), np.full((300, 300, 3), 160, np.uint8))
    assert main(args + ["--reference", str(tmp_path / "blank.png")]) == 2


def test_config_bases_table(tmp_path: Path):
    p = tmp_path / "b.toml"
    head = "[board]\nwidth_mm=1\nheight_mm=1\n[tags]\nids=[0,1,2,3]\nsize_mm=40\n"
    p.write_text(head + "[bases]\ndiameters_mm=[50, 25]\nmin_coverage=0.4\n")
    cfg = BoardConfig.load(p)
    assert cfg.bases_diameters_mm == (25.0, 50.0) and cfg.bases_min_coverage == 0.4
    p.write_text(head + "[bases]\ndiameters_mm=[]\n")
    with pytest.raises(ValueError):
        BoardConfig.load(p)
