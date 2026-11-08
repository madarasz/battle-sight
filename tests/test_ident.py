"""Operative identity (priority 4) on synthetic boards. No camera, no model
download: the tests use the colour-histogram embedder."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from battlesight import ident as idt
from battlesight.app import main
from battlesight.config import BoardConfig
from battlesight.overlay import Menu
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


def test_menu_hit_and_fit():
    m = Menu(["a", "b", "c"], (100, 50), row_h=20, width=80)
    assert m.hit(110, 55) == 0 and m.hit(110, 95) == 2
    assert m.hit(110, 111) is None and m.hit(99, 55) is None
    f = m.fit((150, 60))
    x0, y0, x1, y1 = f.rect()
    assert x1 <= 150 and y0 == 0


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


def test_cli_enroll_images_then_image(tmp_path, capsys):
    from battlesight.config import DEFAULT_CONFIG

    # A board config for the synthetic board size, with the colour embedder.
    text = DEFAULT_CONFIG.read_text()
    text = text.replace("width_mm = 703.0", "width_mm = 606.0").replace("height_mm = 606.0", "height_mm = 703.0", 1)
    text = text.replace("thickness_mm = 1.0", "thickness_mm = 0.0").replace("printed = true", "printed = false")
    text = text.replace('model = "facebook/dinov2-small"', 'model = "color-hist"')
    cfg_file = tmp_path / "board.toml"
    cfg_file.write_text(text)
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


# --- live corner warning (ident and enroll windows) -----------------------------


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


def test_notice_shows_then_expires(capsys):
    from battlesight.app import _Notice
    from battlesight.overlay import GREEN, RED

    now = [100.0]
    n = _Notice(seconds=5, clock=lambda: now[0])
    assert n.current() is None
    n.error("no base found")
    assert n.current() == ("no base found", RED)
    assert "no base found" in capsys.readouterr().err
    n.ok("photo 3 added")
    assert n.current() == ("photo 3 added", GREEN)
    now[0] += 5.1
    assert n.current() is None


def test_strips_under_banner_corner_warning_first():
    from battlesight.app import _CornerCheck, _Notice

    cfg = ident_cfg()
    img = np.full((1080, 1920, 3), 128, np.uint8)
    check = _CornerCheck(cfg)
    n = _Notice()
    assert check.draw(img, n) is img
    n.error("no base found")
    one = check.draw(img, n)
    h = 44  # strip height at 1080p
    assert (one[h + 5, 5] != 128).any() and (one[2 * h + 5, 5] == 128).all()  # notice in row 1
    check.warning = "CORNERS NOT SEEN: 3/4 cards, missing 2 bottom-right"
    two = check.draw(img, n)
    assert (two[2 * h + 5, 5] != 128).any()  # notice moved to row 2


def test_failed_stage_text():
    from battlesight.app import _failed_stage

    cfg = ident_cfg()
    img, mm2px = render_board(cfg)
    p = mm2px((-25, -25)).astype(int)  # tag 0
    cv2.rectangle(img, tuple(p - 60), tuple(p + 60), (160, 160, 160), -1)
    r = Pipeline(cfg).analyze(img)
    assert _failed_stage(r).startswith("corners failed: 3/4 tags")


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
