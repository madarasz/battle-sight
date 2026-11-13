"""Operative identity (priority 4) on synthetic boards. No camera, no model
download: the tests use the colour-histogram embedder."""

from __future__ import annotations

import json
import re

import cv2
import numpy as np
import pytest

from battlesight import ident as idt
from battlesight.app import main
from battlesight.config import BoardConfig
from battlesight.pipeline import Pipeline

from .conftest import Base, make_config, render_board

# Painted models seen from above: (BGR body colour, BGR detail colour).
PAINT = {
    "Nob": ((40, 40, 200), (40, 200, 230)),  # red, yellow detail
    "Grot": ((200, 80, 40), (230, 230, 230)),  # blue, white detail
    "Burna": ((40, 160, 40), (20, 20, 20)),  # green, black detail
}
SIZES = {"Nob": 40.0, "Grot": 25.0, "Burna": 32.0}


def ident_cfg(**over) -> BoardConfig:
    return make_config(ident_model="color-hist", **over)


def paint_board(cfg: BoardConfig, placed: dict[str, tuple[tuple[float, float], float]], px_per_mm: float = 2.0):
    """Board with one painted operative per entry: name -> (centre mm, turn deg)."""
    bases = tuple(Base(c, SIZES[n]) for n, (c, _) in placed.items())
    img, mm2px = render_board(cfg, px_per_mm=px_per_mm, bases=bases)
    for name, (c, turn) in placed.items():
        body, detail = PAINT[name]
        r = 0.5 * SIZES[name]
        p = mm2px(c)
        cv2.circle(img, (int(p[0]), int(p[1])), int((r - 3) * px_per_mm), body, -1, cv2.LINE_AA)
        a = np.deg2rad(turn)
        q = mm2px(np.array(c) + 0.4 * r * np.array([np.cos(a), np.sin(a)]))
        cv2.circle(img, (int(q[0]), int(q[1])), int(0.3 * r * px_per_mm), detail, -1, cv2.LINE_AA)
    return img


def make_gallery(tmp_path, cfg: BoardConfig, teams=("orks",)) -> idt.Gallery:
    return idt.Gallery.for_teams(teams, cfg, tmp_path / "gallery")


def enroll_from(gallery: idt.Gallery, cfg: BoardConfig, img: np.ndarray, names_at: dict[str, tuple[float, float]], team: str = "orks") -> None:
    r = Pipeline(cfg, "ident").analyze(img)
    assert r.ok, r.summary()
    found = r.scratch["bases"]
    for name, c in names_at.items():
        i = int(np.argmin([np.linalg.norm(b.center_mm - c) for b in found]))
        gallery.add(idt.label_of(team, name), r.scratch["base_crops"][i], found[i].diameter_mm)


# --- crops --------------------------------------------------------------------


def test_prepare_size_mask_and_rotation():
    p = idt.IdentParams(input_px=112, crop_pad_mm=20)
    d, ppm = 32.0, 4.0
    img = np.full((800, 800, 3), 60, np.uint8)
    cv2.rectangle(img, (400, 380), (460, 420), (0, 0, 255), -1)
    crop = idt.store_crop(img, (400, 400), d, ppm)
    assert crop.shape == (int((d + 2 * idt.STORE_PAD_MM) * ppm),) * 2 + (3,)
    a = idt.prepare(crop, d, p)
    assert a.shape == (112, 112, 3)
    assert (a[0, 0] == idt.MASK_GRAY).all() and (a[-1, -1] == idt.MASK_GRAY).all()
    # The red bar lies right of the centre; a quarter turn puts it above or below.
    assert a[56, 70, 2] > 200
    b = idt.prepare(crop, d, p, 90.0)
    assert b[56, 70, 2] < 100 and max(b[42, 56, 2], b[70, 56, 2]) > 200


def test_crop_outside_image_is_grey():
    crop = idt.store_crop(np.zeros((100, 100, 3), np.uint8), (5, 5), 25.0, 4.0)
    assert (crop[0, 0] == idt.MASK_GRAY).all() and (crop[-1, -1] == idt.MASK_GRAY).all()


def test_color_embedder_normalised():
    e = idt.ColorEmbedder().embed([np.full((64, 64, 3), v, np.uint8) for v in (10, 200)])
    assert e.shape[0] == 2
    assert np.allclose(np.linalg.norm(e, axis=1), 1.0)


# --- matching -----------------------------------------------------------------


def test_assign_one_to_one_takes_second_choice():
    labels = ["a", "b"]
    scores = np.array([[0.9, 0.6], [0.85, 0.7]])
    out = idt.assign(scores, np.ones_like(scores, bool), labels, min_score=0.5, min_margin=0.03)
    assert [x.label for x in out] == ["a", "b"]
    assert out[0].sure and not out[1].sure  # base 1 got its second choice
    assert out[1].margin < 0
    free = idt.assign(scores, np.ones_like(scores, bool), labels, 0.5, 0.03, unique=False)
    assert [x.label for x in free] == ["a", "a"]


def test_assign_size_limit_and_unknown():
    labels = ["big", "small"]
    scores = np.array([[0.95, 0.9], [0.4, 0.3]])
    allowed = np.array([[False, True], [True, True]])
    out = idt.assign(scores, allowed, labels, 0.5, 0.03)
    assert out[0].label == "small"
    assert [c for c, _ in out[0].candidates] == ["small"]
    assert out[1].label is None  # best allowed score 0.4 < min_score
    assert out[1].to_dict()["margin"] is None


# --- gallery ------------------------------------------------------------------


def test_gallery_team_file_add_reload_remove(tmp_path):
    cfg = ident_cfg()
    g = make_gallery(tmp_path, cfg)
    op = g.ensure_operative("orks", "Boss Nob", 40)
    assert (tmp_path / "gallery" / "orks" / "team.toml").exists()
    with pytest.raises(ValueError, match="40 mm base"):
        g.ensure_operative("orks", "Boss Nob", 32)
    with pytest.raises(ValueError):
        g.ensure_operative("orks", "a/b", 32)
    crop = np.full((480, 480, 3), (40, 40, 200), np.uint8)
    path = g.add(op.label, crop, 40)
    assert path.parent.name == "Boss Nob" and len(g) == 1
    assert g.view_emb.shape[0] == cfg.ident_rotations

    calls = []

    class Counting(idt.ColorEmbedder):
        def embed(self, images):
            calls.append(len(images))
            return super().embed(images)

    g2 = idt.Gallery.for_teams(["orks"], cfg, tmp_path / "gallery", embedder=Counting())
    assert calls == []  # from the cache
    assert g2.operatives[0].base_mm == 40 and g2.counts() == {"orks/Boss Nob": 1}
    g2.remove(path)
    assert len(g2) == 0 and not path.exists()


def test_prepare_averages_pixels_when_it_shrinks():
    crop = ((np.indices((480, 480)).sum(0) % 2) * 255).astype(np.uint8)  # 1 px checker
    crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    p = idt.IdentParams(input_px=112, crop_pad_mm=25)
    for angle in (0.0, 45.0):
        out = idt.prepare(crop, 40.0, p, angle)
        assert out.shape == (112, 112, 3)
        assert out[36:76, 36:76].std() < 25  # no aliasing pattern
    assert "prepare" in p.cache_key("m")


def test_prepare_keeps_the_centre_when_it_shrinks():
    crop = np.full((480, 480, 3), 200, np.uint8)
    cv2.circle(crop, (240, 240), 40, (0, 0, 0), -1)
    out = idt.prepare(crop, 40.0, idt.IdentParams(input_px=112, crop_pad_mm=25), 45.0)
    dark = np.argwhere(out[..., 0] < 60)
    assert np.abs(dark.mean(axis=0) - 55.5).max() < 1.0


def test_unknown_base_size_uses_configured_crop_scale(tmp_path):
    g = make_gallery(tmp_path, ident_cfg(bases_px_per_mm=2.0))
    crop = np.zeros((240, 240, 3), np.uint8)  # (40 + 2 * 40) mm at 2 px per mm
    assert g._diameter(idt.Operative("orks", "Nob", None), crop) == 40.0


def test_two_teams_and_leave_one_out(tmp_path):
    cfg = ident_cfg()
    g = make_gallery(tmp_path, cfg, teams=("orks", "guard"))
    for k in range(3):
        g.add("orks/Nob", np.full((400, 400, 3), (40, 40, 200 - k), np.uint8), 32)
        g.add("guard/Sarge", np.full((400, 400, 3), (200 - k, 80, 40), np.uint8), 32)
    right, total, misses = g.leave_one_out()
    assert (right, total, misses) == (6, 6, [])
    assert sorted(g.labels()) == ["guard/Sarge", "orks/Nob"]
    with pytest.raises(ValueError, match="not loaded"):
        g.ensure_operative("eldar", "x", 25)


# --- pipeline -----------------------------------------------------------------

FIRST = {"Nob": ((150, 150), 0), "Grot": ((300, 200), 90), "Burna": ((420, 500), 200)}
SECOND = {"Nob": ((450, 300), 120), "Grot": ((120, 550), 300), "Burna": ((250, 400), 45)}


def test_ident_stage_names_moved_and_turned_operatives(tmp_path):
    cfg = ident_cfg()
    g = make_gallery(tmp_path, cfg)
    enroll_from(g, cfg, paint_board(cfg, FIRST), {n: c for n, (c, _) in FIRST.items()})
    r = Pipeline(cfg, gallery=g).analyze(paint_board(cfg, SECOND))
    assert r.ok, r.summary()
    st = r.stages["ident"]
    assert st.message.startswith("3/3 named")
    got = {b["label"]: b["center_mm"] for b in st.data["bases"]}
    for name, (c, _) in SECOND.items():
        assert np.linalg.norm(np.subtract(got[f"orks/{name}"], c)) < 2.0
    json.dumps(r.to_dict())


def test_ident_stage_without_gallery_passes():
    cfg = ident_cfg()
    r = Pipeline(cfg).analyze(paint_board(cfg, FIRST))
    assert r.ok and r.stages["ident"].message == "no gallery"
    assert len(r.scratch["base_crops"]) == 3


def test_ident_stage_empty_gallery_passes(tmp_path):
    cfg = ident_cfg()
    r = Pipeline(cfg, gallery=make_gallery(tmp_path, cfg)).analyze(paint_board(cfg, FIRST))
    assert r.ok and "enroll" in r.stages["ident"].message


# --- fixes --------------------------------------------------------------------


def test_label_session_fix_moves_label_and_feeds_gallery(tmp_path):
    cfg = ident_cfg()
    g = make_gallery(tmp_path, cfg)
    g.ensure_operative("orks", "Nob", 32)
    g.ensure_operative("orks", "Kommando", 32)
    crops = [np.full((448, 448, 3), v, np.uint8) for v in (50, 150)]
    idents = [
        idt.Ident("orks/Nob", 0.8, 0.1, [("orks/Nob", 0.8), ("orks/Kommando", 0.7)], sure=True),
        idt.Ident(None, 0.3, -np.inf, [("orks/Kommando", 0.3)]),
    ]
    s = idt.LabelSession(np.array([[100.0, 100.0], [200.0, 100.0]]), [32.0, 32.0], crops, idents, g)
    assert s.pick((300, 300)) is None
    assert s.pick((112, 100)) == 0  # inside the 32 mm disc plus 5 mm
    assert s.menu() == ["orks/Nob", "orks/Kommando"]
    s.pick((200, 100))
    assert s.menu() == ["orks/Kommando", "orks/Nob"]
    s.choose("orks/Nob")
    assert s.idents[1].label == "orks/Nob" and s.idents[1].confirmed
    assert s.idents[0].label is None  # the name moved
    assert g.counts()["orks/Nob"] == 1
    assert s.accept_all() == []  # nothing named and unconfirmed left
    s.undo()
    assert g.counts()["orks/Nob"] == 0 and s.idents[1].label is None


def test_label_session_remove_drops_base_and_its_photos(tmp_path):
    cfg = ident_cfg()
    g = make_gallery(tmp_path, cfg)
    g.ensure_operative("orks", "Nob", 32)
    g.ensure_operative("orks", "Kommando", 32)
    crops = [np.full((448, 448, 3), v, np.uint8) for v in (50, 150, 250)]
    idents = [idt.Ident(None, 0.3, -np.inf, [("orks/Nob", 0.3)]) for _ in crops]
    s = idt.LabelSession(np.array([[100.0, 100.0], [200.0, 100.0], [300.0, 100.0]]), [32.0, 32.0, 32.0], crops, idents, g)
    s.choose("orks/Nob", 1)  # a false base named by mistake
    s.choose("orks/Kommando", 2)
    s.selected = 2
    removed = s.remove(1)
    assert len(removed) == 1 and not removed[0].exists() and g.counts()["orks/Nob"] == 0
    assert len(s.idents) == len(s.crops) == len(s.diameters_mm) == len(s.centers_mm) == 2
    assert s.idents[1].label == "orks/Kommando" and s.selected == 1  # later bases move one down
    assert s.base_at((200, 100)) is None and s.base_at((300, 100)) == 1
    assert s.undo()[0] and s.idents[1].label is None and g.counts()["orks/Kommando"] == 0
    assert s.undo() == (False, None)  # the choice of the removed base is not in the history


# --- done test ----------------------------------------------------------------


def test_score_truth(tmp_path):
    path = tmp_path / "t.ident.json"
    path.write_text(json.dumps({"teams": ["orks"], "tol_mm": 5, "bases": [
        {"center_mm": [100, 100], "label": "orks/A"},
        {"center_mm": [200, 100], "label": "orks/B"},
        {"center_mm": [400, 400], "label": "orks/C"},
    ]}))
    truth = idt.IdentTruth.load(path)
    centers = np.array([[101.0, 100.0], [199.0, 101.0]])
    idents = [idt.Ident("orks/A", 0.9, 0.1), idt.Ident("orks/C", 0.9, 0.1)]
    s = idt.score_truth(truth, centers, idents)
    assert (s["right"], s["total"], s["done_ok"]) == (1, 3, False)
    assert s["wrong"] == [("orks/B", "orks/C")] and s["missed"] == ["orks/C"]


# --- config and CLI -------------------------------------------------------------


def test_config_ident_table(tmp_path):
    from battlesight.config import DEFAULT_CONFIG

    text = DEFAULT_CONFIG.read_text().replace('model = "facebook/dinov2-small"', 'model = "color-hist"').replace("rotations = 8", "rotations = 4")
    path = tmp_path / "board.toml"
    path.write_text(text)
    cfg = BoardConfig.load(path)
    assert cfg.ident_model == "color-hist" and cfg.ident_rotations == 4
    assert idt.IdentParams.from_config(cfg).rotations == 4


def cli_config(tmp_path):
    """A board config file for the synthetic board size, with the colour embedder."""
    from battlesight.config import DEFAULT_CONFIG

    text = DEFAULT_CONFIG.read_text()
    text = text.replace("width_mm = 703.0", "width_mm = 606.0").replace("height_mm = 606.0", "height_mm = 703.0", 1)
    text = text.replace("thickness_mm = 1.0", "thickness_mm = 0.0").replace("printed = true", "printed = false")
    text = text.replace('model = "facebook/dinov2-small"', 'model = "color-hist"')
    cfg_file = tmp_path / "board.toml"
    cfg_file.write_text(text)
    return cfg_file


def test_cli_enroll_images_then_image(tmp_path, capsys):
    cfg_file = cli_config(tmp_path)
    cfg = BoardConfig.load(cfg_file)
    root = tmp_path / "gallery"
    for name, (c, turn) in FIRST.items():
        shot = tmp_path / f"enroll-{name}.png"
        cv2.imwrite(str(shot), paint_board(cfg, {name: (c, turn)}))
        code = main(["enroll", "--team", "orks", "--operative", name, "--images", str(shot), "--gallery-root", str(root), "--config", str(cfg_file)])
        assert code == 0
    assert idt.load_team_file(root / "orks") == SIZES
    game = tmp_path / "game"
    game.mkdir()
    cv2.imwrite(str(game / "g.png"), paint_board(cfg, SECOND))
    capsys.readouterr()
    code = main(["image-pipeline", str(game), "--config", str(cfg_file), "--team", "orks", "--gallery-root", str(root), "--json", "-", "-q"])
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["teams"] == ["orks"]
    labels = sorted(b["label"] for b in report["images"][0]["stages"]["ident"]["data"]["bases"])
    assert labels == ["orks/Burna", "orks/Grot", "orks/Nob"]
    code = main(["gallery", "--team", "orks", "--gallery-root", str(root), "--config", str(cfg_file)])
    assert code == 0
    assert "orks/Nob" in capsys.readouterr().out


def test_leave_one_out_text():
    from battlesight.app import leave_one_out_text

    assert "needs two photos" in leave_one_out_text(0, 0, [])
    assert leave_one_out_text(3, 3, []) == "leave-one-out: 3/3 photos named right (100%)"
    misses = [("orks/A", "orks/B"), ("orks/C", "orks/A"), ("orks/A", "orks/B"), ("orks/D", "orks/A")]
    assert leave_one_out_text(4, 8, misses) == "leave-one-out: 4/8 photos named right (50%): A as B x2, C as A, +1 more"


def test_enroll_live_shows_resolution_and_leave_one_out(tmp_path, monkeypatch):
    import battlesight.app as app

    cfg_file = cli_config(tmp_path)
    cfg = BoardConfig.load(cfg_file)
    root = tmp_path / "gallery"
    g = idt.Gallery.for_teams(("orks",), cfg, root)
    for k in range(3):
        g.add("orks/Nob", np.full((400, 400, 3), (40, 40, 200 - k), np.uint8), 40)
    frame = paint_board(cfg, {"Nob": FIRST["Nob"]})

    class FakeCamera:
        def read(self):
            return frame.copy()

        def release(self):
            pass

    keys = iter([ord("t"), -1, ord(" "), -1, ord("q")])
    frames, current = [], []
    real_banner = app.draw_text_banner

    def banner(img, text, color, extra="", row=0):
        current.append((text, extra))
        real_banner(img, text, color, extra, row)

    def show(_window, _img):
        frames.append(list(current))
        current.clear()

    monkeypatch.setattr(app, "_open_camera", lambda _args: FakeCamera())
    monkeypatch.setattr(app, "draw_text_banner", banner)
    monkeypatch.setattr(cv2, "namedWindow", lambda *a: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda *a: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "imshow", show)
    monkeypatch.setattr(cv2, "waitKey", lambda _ms: next(keys))
    code = main(["enroll", "--team", "orks", "--operative", "Nob", "--gallery-root", str(root), "--config", str(cfg_file), "--reference", str(tmp_path / "none.png")])
    assert code == 0
    size = f"{frame.shape[1]}x{frame.shape[0]}"
    assert frames[0][0][1].startswith(f"{size} -- fps, 3 photos")  # one tick: no rate yet
    assert re.match(rf"{size} \d+ fps, 3 photos", frames[1][0][1])
    assert ("leave-one-out: 3/3 photos named right (100%)", "") in frames[0]
    assert ("leave-one-out: 3/3 photos named right (100%)", "") in frames[1]  # it stays
    assert re.match(rf"{size} \d+ fps, 4 photos", frames[2][0][1])
    assert not any("leave-one-out" in text for text, _ in frames[2])  # gallery changed


# --- live corner warning (ident and enroll windows) -----------------------------


def enroll_loop(monkeypatch, app, frame, steps):
    """Runs the enroll window on `frame`. Each step is a key, or (x, y) for a click.
    Returns the banner lines of each shown image."""
    class FakeCamera:
        def read(self):
            return frame.copy()

        def release(self):
            pass

    steps = iter(steps)
    frames, current, callback = [], [], {}
    real_banner = app.draw_text_banner

    def banner(img, text, color, extra="", row=0):
        current.append((text, extra))
        real_banner(img, text, color, extra, row)

    def show(_window, _img):
        frames.append(list(current))
        current.clear()

    def wait(_ms):
        step = next(steps)
        if isinstance(step, tuple):
            callback["fn"](cv2.EVENT_LBUTTONDOWN, step[0], step[1], 0, None)
            return -1
        return step

    monkeypatch.setattr(app, "_open_camera", lambda _args: FakeCamera())
    monkeypatch.setattr(app, "draw_text_banner", banner)
    monkeypatch.setattr(cv2, "namedWindow", lambda *a: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda _w, fn: callback.update(fn=fn))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "imshow", show)
    monkeypatch.setattr(cv2, "waitKey", wait)
    return frames


def test_enroll_live_view_is_top_down_and_held_while_two_cards_are_covered(tmp_path, monkeypatch):
    import battlesight.app as app
    from battlesight.live import DISPLAY_WIDTH
    from battlesight.overlay import fit_to_width

    from .test_live import _cover_tags

    cfg_file = cli_config(tmp_path)
    cfg = BoardConfig.load(cfg_file)
    frame = paint_board(cfg, {"Nob": FIRST["Nob"]})
    covered = _cover_tags(cfg, frame, (0, 2))
    settle = cfg.settle_frames + 2
    hold = cfg.lost_after_frames + 2  # past the LOST report: the view is held
    frames = iter([frame] * settle + [covered] * hold + [frame, frame])
    shapes = []

    class FakeCamera:
        def read(self):
            return next(frames).copy()

        def release(self):
            pass

    keys = iter([-1] * (settle + hold) + [ord("v"), ord("q")])
    monkeypatch.setattr(app, "_open_camera", lambda _args: FakeCamera())
    monkeypatch.setattr(cv2, "namedWindow", lambda *a: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda *a: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(cv2, "imshow", lambda _w, img: shapes.append(img.shape))
    monkeypatch.setattr(cv2, "waitKey", lambda _ms: next(keys))
    code = main(["enroll", "--team", "orks", "--operative", "Nob", "--gallery-root", str(tmp_path / "gallery"), "--config", str(cfg_file), "--reference", str(tmp_path / "none.png")])
    assert code == 0
    camera = fit_to_width(frame, DISPLAY_WIDTH).shape
    top_down = shapes[0]
    assert top_down != camera  # all 4 cards seen: adjusted perspective at once
    assert set(shapes[settle : settle + hold]) == {top_down}  # two cards covered: the saved position stays
    assert shapes[settle + hold] == camera  # v: the camera view


def test_enroll_relaxed_search_needs_a_click(tmp_path, monkeypatch):
    import battlesight.app as app

    cfg_file = cli_config(tmp_path)
    cfg = BoardConfig.load(cfg_file)
    root = tmp_path / "gallery"
    centre = (300.0, 300.0)
    frame, _ = render_board(cfg, px_per_mm=2.0, bases=(Base(centre, 40.0, model_deg=0.0, model_span_deg=225.0),))
    s, m = cfg.rectify_px_per_mm, cfg.rectify_margin_mm
    click = (int((centre[0] + m) * s), int((centre[1] + m) * s))
    frames = enroll_loop(monkeypatch, app, frame, [ord(" "), -1, click, -1, ord("q")])
    code = main(["enroll", "--team", "orks", "--operative", "Nob", "--base-mm", "40", "--gallery-root", str(root), "--config", str(cfg_file), "--reference", str(tmp_path / "none.png")])
    assert code == 0
    assert frames[0][0][0].startswith("not sure: click the base") and ", 0 photos" in frames[0][0][1]  # no photo yet
    assert len(list((root / "orks" / "Nob").glob("*.png"))) == 1  # the click added it


def test_enroll_esc_leaves_the_base_choice_then_quits(tmp_path, monkeypatch):
    import battlesight.app as app

    cfg_file = cli_config(tmp_path)
    cfg = BoardConfig.load(cfg_file)
    root = tmp_path / "gallery"
    frame, _ = render_board(cfg, px_per_mm=2.0, bases=(Base((260.0, 300.0), 40.0), Base((340.0, 300.0), 40.0)))
    frames = enroll_loop(monkeypatch, app, frame, [ord(" "), -1, 27, -1, 27])
    code = main(["enroll", "--team", "orks", "--operative", "Nob", "--base-mm", "40", "--gallery-root", str(root), "--config", str(cfg_file), "--reference", str(tmp_path / "none.png")])
    assert code == 0
    assert frames[1][0][0] == "click the base of Nob" and "Esc back" in frames[1][0][1]
    assert len(frames) == 4  # the first Esc stays in the app, the second one quits
    assert frames[2][0][0].startswith("enroll Nob:") and frames[2][1][0].startswith("no photo added")
    assert not list((root / "orks" / "Nob").glob("*.png"))


def test_enroll_locate_key_finds_the_base_at_the_click(tmp_path, monkeypatch):
    import battlesight.app as app

    cfg_file = cli_config(tmp_path)
    cfg = BoardConfig.load(cfg_file)
    root = tmp_path / "gallery"
    centre = (300.0, 300.0)
    frame, _ = render_board(cfg, px_per_mm=2.0, bases=(Base(centre, 40.0, model_deg=0.0, model_span_deg=225.0),))
    s, m = cfg.rectify_px_per_mm, cfg.rectify_margin_mm
    click = (int((centre[0] + m) * s), int((centre[1] + m) * s))
    miss = (int((centre[0] + 150 + m) * s), click[1])  # empty board: locate stays on
    frames = enroll_loop(monkeypatch, app, frame, [-1, ord("l"), miss, click, -1, ord("q")])
    code = main(["enroll", "--team", "orks", "--operative", "Nob", "--base-mm", "40", "--gallery-root", str(root), "--config", str(cfg_file), "--reference", str(tmp_path / "none.png")])
    assert code == 0
    assert frames[1][0][0] == "locate Nob: click the centre of its base" and "Esc stop" in frames[1][0][1]
    assert frames[2][0][0].startswith("locate Nob") and frames[2][1][0].startswith("no base found there")
    assert frames[3][0][0].startswith("enroll Nob") and ", 1 photos" in frames[3][0][1]
    assert len(list((root / "orks" / "Nob").glob("*.png"))) == 1  # the click added it


def test_enroll_esc_stops_locate_then_quits(tmp_path, monkeypatch):
    import battlesight.app as app

    cfg_file = cli_config(tmp_path)
    frame, _ = render_board(BoardConfig.load(cfg_file), px_per_mm=2.0)
    frames = enroll_loop(monkeypatch, app, frame, [ord("l"), 27, -1, 27])
    code = main(["enroll", "--team", "orks", "--operative", "Nob", "--gallery-root", str(tmp_path / "gallery"), "--config", str(cfg_file), "--reference", str(tmp_path / "none.png")])
    assert code == 0
    assert len(frames) == 3  # the first Esc stops locate, the second one quits
    assert frames[0][0][0].startswith("locate Nob") and frames[1][0][0].startswith("enroll Nob")
    assert frames[1][1][0] == "locate stopped"


def test_gallery_keeps_background_out_of_photos_and_removes_it(tmp_path):
    cfg = ident_cfg()
    g = make_gallery(tmp_path, cfg)
    g.ensure_operative("orks", "Nob", 32)
    crop, bg = np.full((448, 448, 3), 90, np.uint8), np.full((448, 448, 3), 200, np.uint8)
    path = g.add("orks/Nob", crop, 32.0, bg)
    saved = idt.background_path(path)
    assert saved == path.parent / "bg" / path.name and (cv2.imread(str(saved)) == bg).all()
    assert len(make_gallery(tmp_path, cfg)) == 1  # the reload does not count bg/ as a photo
    g.remove(path)
    assert not path.exists() and not saved.exists()


def test_label_session_saves_background_and_undo_removes_it(tmp_path):
    g = make_gallery(tmp_path, ident_cfg())
    g.ensure_operative("orks", "Nob", 32)
    crop, bg = np.full((448, 448, 3), 90, np.uint8), np.full((448, 448, 3), 200, np.uint8)
    none = idt.Ident(None, float("-inf"), float("-inf"))
    s = idt.LabelSession(np.array([[100.0, 100.0]]), [32.0], [crop], [none], g, backgrounds=[bg])
    path = s.choose("orks/Nob", 0)
    assert idt.background_path(path).exists()
    s.undo()
    assert not idt.background_path(path).exists()


def mask_scene(board=(40, 120, 40), neighbour_mm=None):
    """Stored crop of a 32 mm base at 4 px per mm, its empty board, and the board colour."""
    side = int((32 + 2 * idt.STORE_PAD_MM) * 4)
    bg = np.full((side, side, 3), board, np.uint8)
    crop = bg.copy()
    c = side // 2
    cv2.circle(crop, (c, c), 64, (0, 0, 230), -1)
    cv2.ellipse(crop, (c + 80, c), (40, 16), 0, 0, 360, (0, 200, 230), -1)  # the body leans out
    cv2.circle(crop, (20, 20), 14, (230, 0, 0), -1)  # a token, not joined to the base
    if neighbour_mm is not None:
        cv2.circle(crop, (c + int(neighbour_mm * 4), c), 64, (230, 230, 0), -1)
    return crop, bg, c


def test_mask_keeps_the_operative_and_greys_the_rest():
    p = idt.IdentParams(crop_pad_mm=25)
    crop, bg, c = mask_scene()
    out = idt.mask_crop(crop, bg, 32.0, p)
    assert out is not None
    assert (out[c, c] == (0, 0, 230)).all() and (out[c, c + 100] == (0, 200, 230)).all()  # base and body
    assert (out[20, 20] == idt.MASK_GRAY).all()  # the token
    assert (out[c + 150, c] == idt.MASK_GRAY).all()  # the board


def test_mask_cuts_off_a_touching_neighbour():
    p = idt.IdentParams(crop_pad_mm=25)
    crop, bg, c = mask_scene(neighbour_mm=34.0)
    out = idt.mask_crop(crop, bg, 32.0, p, [(34.0, 0.0, 32.0)])
    assert out is not None and (out[c, c] == (0, 0, 230)).all()
    assert (out[c, c + 136] == idt.MASK_GRAY).all()


def test_mask_refuses_a_wrong_background():
    p = idt.IdentParams(crop_pad_mm=25)
    crop, bg, _ = mask_scene()
    assert idt.mask_crop(crop, crop.copy(), 32.0, p) is None  # the operative is in the "empty" photo
    brighter = np.clip(crop.astype(int) + 90, 0, 255).astype(np.uint8)
    assert idt.mask_crop(brighter, bg, 32.0, p) is None  # the light changed: all of the crop differs
    assert idt.mask_crop(crop, np.full_like(crop, 240), 32.0, p) is None  # another board


def test_mask_off_keeps_gallery_as_before(tmp_path):
    g = make_gallery(tmp_path, ident_cfg())
    g.ensure_operative("orks", "Nob", 32)
    crop, bg, _ = mask_scene()
    g.add("orks/Nob", crop, 32.0, bg)
    assert not g.can_mask and len(g.masked_emb) == 0 and g.view_emb.shape[0] == g.params.rotations


def test_gallery_masked_views_cache_and_missing_background(tmp_path):
    cfg = ident_cfg(ident_mask=True)
    g = make_gallery(tmp_path, cfg)
    g.ensure_operative("orks", "Nob", 32)
    g.ensure_operative("orks", "Grot", 32)
    crop, bg, _ = mask_scene()
    g.add("orks/Nob", crop, 32.0, bg)
    assert g.view_emb.shape[0] == g.params.rotations and g.masked_emb.shape == g.view_emb.shape
    assert g.can_mask  # Grot has no photos: nothing to compare
    no_bg = g.add("orks/Grot", crop, 32.0)
    assert g.view_emb.shape[0] == 2 * g.params.rotations and g.masked_emb.shape[0] == g.params.rotations
    assert not g.can_mask  # Grot has photos, none masked

    class NoEmbed(idt.ColorEmbedder):
        def embed(self, images):
            raise AssertionError("the cache must hold the embeddings")

    again = idt.Gallery.for_teams(("orks",), cfg, tmp_path / "gallery", NoEmbed())
    assert again.masked_emb.shape == g.masked_emb.shape and not again.can_mask
    g.remove(no_bg)
    assert g.can_mask
    right, total, misses = g.leave_one_out()
    assert (right, total, misses) == (0, 0, [])  # one photo: nothing to compare with


def test_identify_with_mask_ignores_the_board_colour(tmp_path):
    cfg = ident_cfg(ident_mask=True)
    g = make_gallery(tmp_path, cfg)
    g.ensure_operative("orks", "Nob", 32)
    crop, bg, _ = mask_scene()
    g.add("orks/Nob", crop, 32.0, bg)
    game, game_bg, _ = mask_scene(board=(150, 60, 60))
    masked = idt.identify([game], [32.0], g, [game_bg], [(100.0, 100.0)])[0]
    full = idt.identify([game], [32.0], g)[0]
    assert masked.masked and not full.masked
    assert masked.score > 0.9 and masked.score > full.score + 0.1  # a thin edge of board stays in the mask


def test_ident_stage_masks_with_reference_and_reports_it(tmp_path):
    cfg = ident_cfg(ident_mask=True)
    g = make_gallery(tmp_path, cfg)
    empty = paint_board(cfg, {})
    first = Pipeline(cfg, reference=empty).analyze(paint_board(cfg, FIRST))
    for c, crop, bg in zip(first.scratch["bases"], first.scratch["base_crops"], first.scratch["base_backgrounds"]):
        name = min(FIRST, key=lambda n: np.linalg.norm(np.subtract(FIRST[n][0], c.center_mm)))
        g.ensure_operative("orks", name, SIZES[name])
        g.add(f"orks/{name}", crop, c.diameter_mm, bg)
    r = Pipeline(cfg, reference=empty, gallery=g).analyze(paint_board(cfg, SECOND))
    st = r.stages["ident"]
    assert r.ok and st.data["masked"] == 3 and st.message == "3/3 named, 3 masked"
    got = {b["label"]: b["center_mm"] for b in st.data["bases"]}
    for name, (c, _) in SECOND.items():
        assert np.linalg.norm(np.subtract(got[f"orks/{name}"], c)) < 2.0
    without = Pipeline(cfg, gallery=g).analyze(paint_board(cfg, SECOND))
    assert without.ok and without.stages["ident"].data["masked"] == 0


def test_ident_stage_says_when_gallery_cannot_mask(tmp_path):
    cfg = ident_cfg(ident_mask=True)
    g = make_gallery(tmp_path, cfg)
    enroll_from(g, cfg, paint_board(cfg, FIRST), {n: c for n, (c, _) in FIRST.items()})
    r = Pipeline(cfg, reference=paint_board(cfg, {}), gallery=g).analyze(paint_board(cfg, SECOND))
    assert r.ok and "no mask" in r.stages["ident"].message and r.stages["ident"].data["masked"] == 0


def test_ident_stage_cuts_backgrounds_only_with_reference():
    cfg = ident_cfg()
    empty = paint_board(cfg, {})
    game = paint_board(cfg, {"Nob": FIRST["Nob"]})
    without = Pipeline(cfg).analyze(game)
    assert without.ok and without.scratch["base_backgrounds"] == []
    result = Pipeline(cfg, reference=empty).analyze(game)
    assert result.ok and len(result.scratch["bases"]) == 1
    crop, bg = result.scratch["base_crops"][0], result.scratch["base_backgrounds"][0]
    assert bg.shape == crop.shape
    # The same square of the board: the base is in the crop only.
    centre = crop.shape[0] // 2
    assert np.abs(crop[centre - 40 : centre + 40].astype(int) - bg[centre - 40 : centre + 40].astype(int)).mean() > 5.0


def test_enroll_saves_background_with_reference_and_warns_without(tmp_path, monkeypatch):
    import battlesight.app as app

    cfg_file = cli_config(tmp_path)
    cfg = BoardConfig.load(cfg_file)
    frame = paint_board(cfg, {"Nob": FIRST["Nob"]})
    ref = tmp_path / "empty.png"
    cv2.imwrite(str(ref), paint_board(cfg, {}))
    args = ["enroll", "--team", "orks", "--operative", "Nob", "--base-mm", "40", "--config", str(cfg_file)]

    root = tmp_path / "with"
    frames = enroll_loop(monkeypatch, app, frame, [ord(" "), -1, ord("q")])
    assert main([*args, "--gallery-root", str(root), "--reference", str(ref)]) == 0
    assert "push space" in frames[0][0][0]
    photos = list((root / "orks" / "Nob").glob("*.png"))
    assert len(photos) == 1 and idt.background_path(photos[0]).exists()

    root = tmp_path / "without"
    frames = enroll_loop(monkeypatch, app, frame, [-1, ord("q")])
    assert main([*args, "--gallery-root", str(root), "--reference", str(tmp_path / "none.png")]) == 0
    assert "no background saved" in frames[0][0][0]


def test_enroll_e_key_saves_reference_and_next_grab_uses_it(tmp_path, monkeypatch):
    import battlesight.app as app

    cfg_file = cli_config(tmp_path)
    cfg = BoardConfig.load(cfg_file)
    root = tmp_path / "gallery"
    ref = tmp_path / "sub" / "empty.png"
    frame, _ = render_board(cfg, px_per_mm=2.0)
    has_reference = []
    real = app.Pipeline.analyze

    def spy(self, *a, **k):
        has_reference.append(self.has_reference)
        return real(self, *a, **k)

    monkeypatch.setattr(app.Pipeline, "analyze", spy)
    enroll_loop(monkeypatch, app, frame, [ord("e"), ord(" "), ord("q")])
    code = main(["enroll", "--team", "orks", "--operative", "Nob", "--gallery-root", str(root), "--config", str(cfg_file), "--reference", str(ref)])
    assert code == 0
    saved = cv2.imread(str(ref), cv2.IMREAD_COLOR)
    assert saved is not None and saved.shape == frame.shape
    assert has_reference[-1] is True


def test_enroll_e_key_without_board_writes_nothing(tmp_path, monkeypatch):
    import battlesight.app as app

    cfg_file = cli_config(tmp_path)
    ref = tmp_path / "empty.png"
    frame = np.full((1080, 1920, 3), 200, np.uint8)
    enroll_loop(monkeypatch, app, frame, [ord("e"), -1, ord("q")])
    code = main(["enroll", "--team", "orks", "--operative", "Nob", "--gallery-root", str(tmp_path / "gallery"), "--config", str(cfg_file), "--reference", str(ref)])
    assert code == 0
    assert not ref.exists()


def test_corner_warning_text():
    from battlesight.app import corner_warning

    cfg = ident_cfg()
    assert corner_warning(cfg, {0: 1, 1: 1, 2: 1, 3: 1}) is None
    assert corner_warning(cfg, {0: 1, 1: 1, 3: 1}) == "CORNERS NOT SEEN: 3/4 cards, missing 2 bottom-right"


def test_corner_check_on_live_frames():
    from battlesight.app import _CornerCheck

    cfg = ident_cfg()
    img, mm2px = render_board(cfg)
    check = _CornerCheck(cfg, every=2)
    assert check.update(img) is None
    assert check.draw(img) is img  # nothing drawn, no copy
    covered = img.copy()
    p = mm2px((cfg.width_mm + 25, cfg.height_mm + 25)).astype(int)  # tag 2, outside the bottom-right corner
    cv2.rectangle(covered, tuple(p - 60), tuple(p + 60), (160, 160, 160), -1)
    check.update(covered)  # frame 1: not checked (every 2)
    assert check.warning is None
    assert "missing 2 bottom-right" in check.update(covered)
    out = check.draw(covered)
    assert out is not covered and not np.array_equal(out, covered)


def test_fps_meter_smooths_and_resets():
    from battlesight.live import FpsMeter

    m = FpsMeter()
    m.tick(10.0)
    assert m.fps is None
    m.tick(10.1)
    assert m.fps == pytest.approx(10.0)
    m.tick(10.15)
    assert m.fps == pytest.approx(0.9 * 10.0 + 0.1 * 20.0)
    m.reset()
    m.tick(11.0)
    assert m.fps is None


def test_notice_shows_then_expires(capsys):
    from battlesight.live import Notice
    from battlesight.overlay import GREEN, RED

    now = [100.0]
    n = Notice(seconds=5, clock=lambda: now[0])
    assert n.current() is None
    n.error("no base found")
    assert n.current() == ("no base found", RED)
    assert "no base found" in capsys.readouterr().err
    n.ok("photo 3 added")
    assert n.current() == ("photo 3 added", GREEN)
    now[0] += 5.1
    assert n.current() is None


def test_strips_under_banner_corner_warning_first():
    from battlesight.app import _CornerCheck
    from battlesight.live import Notice

    cfg = ident_cfg()
    img = np.full((1080, 1920, 3), 128, np.uint8)
    check = _CornerCheck(cfg)
    n = Notice()
    assert check.draw(img, n) is img
    n.error("no base found")
    one = check.draw(img, n)
    h = 44  # strip height at 1080p
    assert (one[h + 5, 5] != 128).any() and (one[2 * h + 5, 5] == 128).all()  # notice in row 1
    check.warning = "CORNERS NOT SEEN: 3/4 cards, missing 2 bottom-right"
    two = check.draw(img, n)
    assert (two[2 * h + 5, 5] != 128).any()  # notice moved to row 2


def test_failed_stage_text():
    from battlesight.live import failed_stage

    cfg = ident_cfg()
    img, mm2px = render_board(cfg)
    p = mm2px((-25, -25)).astype(int)  # tag 0
    cv2.rectangle(img, tuple(p - 60), tuple(p + 60), (160, 160, 160), -1)
    r = Pipeline(cfg).analyze(img)
    assert failed_stage(r).startswith("corners failed: 3/4 tags")


def test_list_teams(tmp_path):
    root = tmp_path / "g"
    assert idt.list_teams(root) == []
    (root / "orks" / "Nob").mkdir(parents=True)
    (root / "guard").mkdir()
    idt.save_team_file(root / "guard", "guard", {"Sarge": 32})
    (root / "empty").mkdir()
    (root / ".hidden" / "x").mkdir(parents=True)
    assert idt.list_teams(root) == ["guard", "orks"]


def test_cli_image_uses_all_gallery_teams_by_default(tmp_path, capsys):
    from battlesight.config import DEFAULT_CONFIG

    text = DEFAULT_CONFIG.read_text()
    text = text.replace("width_mm = 703.0", "width_mm = 606.0").replace("height_mm = 606.0", "height_mm = 703.0", 1)
    text = text.replace("thickness_mm = 1.0", "thickness_mm = 0.0").replace("printed = true", "printed = false")
    text = text.replace('model = "facebook/dinov2-small"', 'model = "color-hist"')
    cfg_file = tmp_path / "board.toml"
    cfg_file.write_text(text)
    cfg = BoardConfig.load(cfg_file)
    g = idt.Gallery.for_teams(["orks"], cfg, idt.DEFAULT_GALLERY_ROOT)  # the test gallery root (conftest)
    enroll_from(g, cfg, paint_board(cfg, FIRST), {n: c for n, (c, _) in FIRST.items()})
    game = tmp_path / "game.png"
    cv2.imwrite(str(game), paint_board(cfg, SECOND))
    capsys.readouterr()
    assert main(["image-pipeline", str(game), "--config", str(cfg_file), "--json", "-", "-q"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["teams"] == ["orks"]
    assert report["images"][0]["stages"]["ident"]["message"].startswith("3/3 named")
    empty = tmp_path / "no_teams"
    empty.mkdir()
    assert main(["image-pipeline", str(game), "--config", str(cfg_file), "--json", "-", "-q", "--gallery-root", str(empty)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["teams"] is None and report["images"][0]["stages"]["ident"]["message"] == "no gallery"


# --- base sizes of the teams (roster) ------------------------------------------


def test_roster_sizes_from_team_files(tmp_path):
    cfg = ident_cfg()
    root = tmp_path / "gallery"
    assert idt.roster_sizes(root, ["orks"]) is None  # no team file
    g = make_gallery(tmp_path, cfg)
    g.ensure_operative("orks", "Nob", 40)
    g.ensure_operative("orks", "Grot", 25)
    g.ensure_operative("orks", "Boy", 25)
    assert idt.roster_sizes(root, ["orks"]) == (25.0, 40.0) == g.base_sizes()
    g.ensure_operative("orks", "New", None)  # a size that is not known: all sizes stay possible
    assert idt.roster_sizes(root, ["orks"]) is None and g.base_sizes() is None


def test_bases_stage_uses_the_gallery_sizes(tmp_path):
    cfg = ident_cfg()
    img, _ = render_board(cfg, px_per_mm=2.0, bases=(Base((200.0, 200.0), 40.0), Base((450.0, 400.0), 50.0)))
    g = make_gallery(tmp_path, cfg)
    g.ensure_operative("orks", "Nob", 40)
    found = Pipeline(cfg, "bases", gallery=g).analyze(img).scratch["bases"]
    assert [c.diameter_mm for c in found] == [40.0]


# --- the same photo twice -------------------------------------------------------


def test_gallery_refuses_the_same_photo(tmp_path):
    cfg = ident_cfg()
    g = make_gallery(tmp_path, cfg)
    crop = np.full((448, 448, 3), 90, np.uint8)
    cv2.circle(crop, (224, 224), 60, (0, 0, 200), -1)
    first = g.add("orks/Nob", crop, 32)
    with pytest.raises(idt.DuplicatePhoto) as exc:
        g.add("orks/Nob", crop.copy(), 32)
    assert exc.value.path == first and g.counts()["orks/Nob"] == 1
    assert len(list(first.parent.glob("*.png"))) == 1
    other = crop.copy()
    cv2.circle(other, (224, 224), 60, (0, 200, 0), -1)  # another photo
    g.add("orks/Nob", other, 32)
    g.add("orks/Grot", crop, 32)  # the same crop under another name is a new name for it, not a copy
    assert g.counts() == {"orks/Grot": 1, "orks/Nob": 2}


def test_fix_with_a_photo_the_gallery_has_sets_the_name_only(tmp_path):
    cfg = ident_cfg()
    g = make_gallery(tmp_path, cfg)
    crop = np.full((448, 448, 3), 50, np.uint8)
    kept = g.add("orks/Nob", crop, 32)
    idents = [idt.Ident("orks/Nob", 0.9, 0.2, [("orks/Nob", 0.9)], sure=True)]
    s = idt.LabelSession(np.array([[100.0, 100.0]]), [32.0], [crop], idents, g)
    assert s.accept_all() == [None]
    assert s.idents[0].confirmed and g.counts()["orks/Nob"] == 1
    assert s.undo() == (True, None)  # the undo keeps the photo that was there before
    assert kept.is_file() and not s.idents[0].confirmed
    assert s.undo() == (False, None)


def test_weak_operatives_and_advice():
    from battlesight.app import weak_operatives, weak_text

    counts = {"T/A": 20, "T/B": 30, "T/C": 10, "T/D": 10}
    misses = [("T/A", "T/B")] * 5 + [("T/B", "T/A")] * 2 + [("T/C", "T/A")] + [("T/D", "T/C")] * 4
    weak = weak_operatives(misses, counts)
    # B: 2 of 30 is under the rate. C: one wrong photo is not enough. Worst rate first.
    assert weak == [("T/D", 4, 10), ("T/A", 5, 20)]
    assert weak_text(weak) == "add photos of D, A"
    assert weak_text([]) == "" and weak_operatives([], counts) == []
    assert weak_text([("T/A", 2, 5), ("T/B", 2, 5), ("T/C", 2, 5), ("T/D", 2, 5)]) == "add photos of A, B, C, +1 more"


# --- maximum wounds -------------------------------------------------------------


def test_team_file_keeps_the_maximum_wounds(tmp_path):
    team = tmp_path / "orks"
    idt.save_team_file(team, "orks", {"Nob": 40.0, "Grot": 25.0}, {"Nob": 12})
    assert idt.read_team_file(team) == {"Nob": (40.0, 12), "Grot": (25.0, None)}
    assert idt.load_team_file(team) == {"Nob": 40.0, "Grot": 25.0}
    assert [(op.name, op.wounds) for op in idt.team_operatives("orks", team)] == [("Nob", 12), ("Grot", None)]


def test_team_operatives_keep_the_file_order(tmp_path):
    team = tmp_path / "orks"
    idt.save_team_file(team, "orks", {"Nob": 40.0, "Grot": 25.0, "Burna": 32.0})
    for name in ("Zog", "Grot", "Ark"):  # folders: the unlisted ones come last, sorted
        (team / name).mkdir()
    assert [op.name for op in idt.team_operatives("orks", team)] == ["Nob", "Grot", "Burna", "Ark", "Zog"]
    assert list(idt.load_team_file(team)) == ["Nob", "Grot", "Burna"]


@pytest.mark.parametrize("value", ["0", "-2", '"ten"', "1.5", "true"])
def test_team_file_with_a_bad_wounds_value_is_an_error(tmp_path, value):
    (tmp_path / "orks").mkdir()
    (tmp_path / "orks" / "team.toml").write_text(f'[operatives."Nob"]\nbase_mm = 40\nwounds = {value}\n')
    with pytest.raises(ValueError, match="wounds"):
        idt.read_team_file(tmp_path / "orks")


def test_ensure_operative_sets_and_replaces_the_wounds(tmp_path):
    cfg = ident_cfg()
    g = make_gallery(tmp_path, cfg)
    g.ensure_operative("orks", "Nob", 40, 12)
    g.ensure_operative("orks", "Grot", 25)  # a new operative keeps the wounds of the others
    assert idt.read_team_file(tmp_path / "gallery" / "orks")["Nob"] == (40.0, 12)
    assert g.ensure_operative("orks", "Nob", None, 14).wounds == 14
    assert g.ensure_operative("orks", "Nob", None).wounds == 14  # no value: no change
    again = make_gallery(tmp_path, cfg)
    assert again.operatives[again.index("orks/Nob")].wounds == 14
    with pytest.raises(ValueError, match="wounds"):
        g.ensure_operative("orks", "Nob", None, 0)


def test_cli_enroll_wounds_is_saved_at_the_start(tmp_path, capsys):
    cfg_file = cli_config(tmp_path)
    cfg = BoardConfig.load(cfg_file)
    root = tmp_path / "gallery"
    shot = tmp_path / "nob.png"
    cv2.imwrite(str(shot), paint_board(cfg, {"Nob": ((300.0, 300.0), 0.0)}))
    args = ["enroll", "--team", "orks", "--operative", "Nob", "--gallery-root", str(root), "--config", str(cfg_file)]
    assert main([*args, "--wounds", "12", "--images", str(shot)]) == 0
    assert idt.read_team_file(root / "orks")["Nob"] == (40.0, 12)
    # A new value replaces the old one, also when no photo is added.
    assert main([*args, "--wounds", "13", "--images", str(tmp_path / "missing.png")]) == 2
    assert idt.read_team_file(root / "orks")["Nob"] == (40.0, 13)
    assert "maximum wounds 13" in capsys.readouterr().out
    assert main([*args, "--wounds", "0", "--images", str(shot)]) == 2
    assert idt.read_team_file(root / "orks")["Nob"] == (40.0, 13)


def test_team_info_roundtrip_and_keep(tmp_path):
    team = tmp_path / "orks"
    idt.save_team_file(team, "orks", {"Nob": 40.0}, {"Nob": 12}, idt.TeamInfo(("Recon", "Security"), "Waaagh"))
    assert idt.read_team_info(team) == idt.TeamInfo(("Recon", "Security"), "Waaagh")
    # a rewrite without info keeps the team values
    idt.save_team_file(team, "orks", {"Nob": 40.0}, {"Nob": 13})
    assert idt.read_team_info(team) == idt.TeamInfo(("Recon", "Security"), "Waaagh")
    assert idt.read_team_file(team)["Nob"] == (40.0, 13)


def test_team_info_bad_value(tmp_path):
    team = tmp_path / "orks"
    team.mkdir()
    (team / "team.toml").write_text('archetype = ["Bogus"]\n')
    with pytest.raises(ValueError):
        idt.read_team_info(team)


def test_cli_enroll_team_info_replaces_values(tmp_path, capsys):
    cfg_file = cli_config(tmp_path)
    cfg = BoardConfig.load(cfg_file)
    root = tmp_path / "gallery"
    shot = tmp_path / "nob.png"
    cv2.imwrite(str(shot), paint_board(cfg, {"Nob": ((300.0, 300.0), 0.0)}))
    args = ["enroll", "--team", "orks", "--operative", "Nob", "--gallery-root", str(root), "--config", str(cfg_file), "--images", str(shot)]
    missing = [*args[:-2], "--images", str(tmp_path / "missing.png")]  # no photo added: exit 2, values still saved
    assert main([*args, "--archetype", "Seek and Destroy", "Security", "--faction-counter", "Waaagh"]) == 0
    assert idt.read_team_info(root / "orks") == idt.TeamInfo(("Seek and Destroy", "Security"), "Waaagh")
    assert main([*missing, "--archetype", "Recon"]) == 2  # replaces the list, keeps the counter
    assert idt.read_team_info(root / "orks") == idt.TeamInfo(("Recon",), "Waaagh")
    assert main([*missing, "--faction-counter", ""]) == 2  # clears the counter
    assert idt.read_team_info(root / "orks") == idt.TeamInfo(("Recon",), None)
    assert idt.read_team_file(root / "orks")["Nob"][0] == 40.0
    with pytest.raises(SystemExit):
        main([*missing, "--archetype", "Bogus"])


def test_parse_count_range():
    assert idt.parse_count_range(5, "x") == (5, 5)
    assert idt.parse_count_range("5", "x") == (5, 5)
    assert idt.parse_count_range("4-6", "x") == (4, 6)
    assert idt.parse_count_range("4..6", "x") == (4, 6)
    for bad in (0, "0", "6-4", "4-", "-4", "a", "", True, 2.5, [4]):
        with pytest.raises(ValueError):
            idt.parse_count_range(bad, "x")


def test_operative_count_roundtrip_and_cli(tmp_path):
    team = tmp_path / "orks"
    idt.save_team_file(team, "orks", {"Nob": 40.0}, None, idt.TeamInfo(operative_count=(4, 6)))
    assert 'operative-count = "4-6"' in (team / "team.toml").read_text()
    assert idt.read_team_info(team).operative_count == (4, 6)
    idt.save_team_file(team, "orks", {"Nob": 40.0}, None, idt.TeamInfo(operative_count=(5, 5)))
    assert "operative-count = 5\n" in (team / "team.toml").read_text()
    cfg_file = cli_config(tmp_path)
    root = tmp_path / "gallery"
    args = ["enroll", "--team", "orks", "--operative", "Nob", "--gallery-root", str(root), "--config", str(cfg_file), "--images", str(tmp_path / "missing.png")]
    assert main([*args, "--operative-count", "4-6"]) == 2  # no photo: exit 2, value saved
    assert idt.read_team_info(root / "orks").operative_count == (4, 6)
    assert main([*args, "--operative-count", "5"]) == 2
    assert idt.read_team_info(root / "orks").operative_count == (5, 5)
    assert main([*args, "--operative-count", ""]) == 2
    assert idt.read_team_info(root / "orks").operative_count is None
    assert main([*args, "--operative-count", "9-3"]) == 2


def test_expendable_roundtrip_keep_and_cli(tmp_path):
    team = tmp_path / "orks"
    idt.save_team_file(team, "orks", {"Nob": 40.0, "Grot": 25.0}, {"Nob": 12}, None, ["Grot"])
    assert "expendable = true" in (team / "team.toml").read_text()
    assert idt.read_expendable(team) == {"Grot"}
    ops = {o.name: o.expendable for o in idt.team_operatives("orks", team)}
    assert ops == {"Grot": True, "Nob": False}
    idt.save_team_file(team, "orks", {"Nob": 40.0, "Grot": 25.0})  # no value given: the file keeps it
    assert idt.read_expendable(team) == {"Grot"}
    (team / "team.toml").write_text('[operatives."Grot"]\nexpendable = "yes"\n')
    with pytest.raises(ValueError, match="expendable"):
        idt.read_expendable(team)

    cfg_file = cli_config(tmp_path)
    root = tmp_path / "gallery"
    args = ["enroll", "--team", "orks", "--operative", "Grot", "--gallery-root", str(root), "--config", str(cfg_file), "--images", str(tmp_path / "missing.png")]
    assert main([*args, "--expendable"]) == 2  # no photo: exit 2, value saved
    assert idt.read_expendable(root / "orks") == {"Grot"}
    assert main([*args, "--wounds", "5"]) == 2  # no option: the value stays
    assert idt.read_expendable(root / "orks") == {"Grot"}
    assert main([*args, "--not-expendable"]) == 2
    assert idt.read_expendable(root / "orks") == set()


# --- copies of one operative -------------------------------------------------------


def test_copy_labels():
    assert idt.copy_label("orks/Boy", 2) == "orks/Boy#2"
    assert idt.type_of("orks/Boy#2") == "orks/Boy" and idt.type_of("orks/Boy") == "orks/Boy"
    assert idt.copy_number("orks/Boy#2") == 2 and idt.copy_number("orks/Boy") is None
    assert idt.display_name("orks/Boy#2") == "Boy 2" and idt.display_name("orks/Nob") == "Nob"
    op = idt.Operative("orks", "Boy", 32.0, 10, max=3, copy=2)
    assert op.label == "orks/Boy#2" and op.type_label == "orks/Boy" and op.display == "Boy 2"
    assert idt.Operative("orks", "Nob").label == "orks/Nob"
    with pytest.raises(ValueError, match="#"):
        idt.check_name("Boy#1", "operative")


def test_team_file_max_copies(tmp_path):
    team = tmp_path / "orks"
    idt.save_team_file(team, "orks", {"Nob": 40.0, "Boy": 32.0}, {"Boy": 10}, None, None, {"Boy": 3, "Nob": 1})
    text = (team / idt.TEAM_FILE).read_text()
    assert "max = 3" in text and text.count("max =") == 1 and "# max:" in text
    assert idt.read_copies(team) == {"Boy": 3}
    ops = {op.name: op for op in idt.team_operatives("orks", team)}
    assert ops["Boy"].max == 3 and ops["Nob"].max == 1
    idt.save_team_file(team, "orks", {"Nob": 40.0, "Boy": 32.0}, {"Boy": 11})  # no copies given: the file keeps them
    assert idt.read_copies(team) == {"Boy": 3}
    for bad in ("0", "1.5", "true", '"2"'):
        (team / idt.TEAM_FILE).write_text(f'[operatives."Boy"]\nmax = {bad}\n')
        with pytest.raises(ValueError, match="max"):
            idt.read_copies(team)
    (team / idt.TEAM_FILE).write_text('[operatives."Boy#1"]\nbase_mm = 32\n')
    with pytest.raises(ValueError, match="#"):
        idt.read_team_file(team)


def test_gallery_keeps_max_when_it_writes_the_team_file(tmp_path):
    cfg = ident_cfg()
    team = tmp_path / "gallery" / "orks"
    idt.save_team_file(team, "orks", {"Boy": 32.0}, None, None, None, {"Boy": 4})
    g = make_gallery(tmp_path, cfg)
    g.ensure_operative("orks", "Nob", 40.0)  # a new operative writes the file again
    assert idt.read_copies(team) == {"Boy": 4}
    assert g.columns() == [0, 0, 0, 0, 1]
    g.copies = {"orks/Boy": 2}
    assert g.columns() == [0, 0, 1]


def test_assign_copy_columns_margin_against_other_types():
    labels = ["t/Boy", "t/Boy", "t/Nob"]
    scores = np.array([[0.9, 0.9, 0.6], [0.8, 0.8, 0.75], [0.7, 0.7, 0.2]])
    allowed = np.ones_like(scores, bool)
    out = idt.assign(scores, allowed, labels, min_score=0.5, min_margin=0.1)
    got = [x.label for x in out]
    assert got.count("t/Boy") == 2 and got.count("t/Nob") == 1  # two copies: at most two bases
    first = out[0]
    assert first.label == "t/Boy" and first.margin == pytest.approx(0.3) and first.sure  # not 0 against its own copy
    assert [lab for lab, _ in first.candidates] == ["t/Boy", "t/Nob"]  # each label once
