"""Terrain map files (terrain.py), the overlay and the map command."""

from __future__ import annotations

import numpy as np
import pytest

from battlesight import terrain as ter
from battlesight.app import main
from battlesight.overlay import draw_terrain

from .conftest import render_board, tomb_world_grid

CARD_6 = ter.DEFAULT_MAPS_ROOT / "tomb-world" / "layout-6.toml"
PITCH = 97.0


def mm(col, row):
    return 12.0 + PITCH * col, 12.0 + PITCH * row


def parse(text: str, cfg=None):
    import tomllib

    return ter.parse_map(tomllib.loads(text), cfg or tomb_world_grid())


HEAD = 'schema = 1\nname = "T"\nterrain = "Tomb World"\n'


def wall_map(frm, to, pillars, opening=False, extra="", key=None, at=None):
    p = ", ".join(f"[{a}, {b}]" for a, b in pillars)
    key = key or ("A3" if abs(to[0] - frm[0]) + abs(to[1] - frm[1]) == 2 else "B3")
    text = f'{HEAD}pillars = [{p}]\n{extra}\n[[wall]]\nkey = "{key}"\nfrom = [{frm[0]}, {frm[1]}]\nto = [{to[0]}, {to[1]}]\nopening = {str(opening).lower()}\n'
    return text + (f"opening_at = [{at[0]}, {at[1]}]\n" if at else "")


def by_kind(shapes, kind):
    return [s for s in shapes if s.kind == kind]


def size_along(shape, axis):
    return float(np.ptp(shape.points[:, axis]))


def test_card_6_loads():
    m = ter.load_map(CARD_6, tomb_world_grid())
    openings = [w.opening for w in m.walls]
    assert (len(m.walls), openings.count("breach"), openings.count("hatchway")) == (16, 4, 6)
    assert (len(m.objectives), len(m.teleporters), len(m.zones), len(m.wall_ends), len(m.pillars)) == (3, 2, 2, 1, 22)
    assert m.terrain == "Tomb World" and m.name == "Layout 6"


def test_grid_point_to_mm_and_zone_to_edge():
    cfg = tomb_world_grid()
    m = parse(HEAD + '[[objective]]\nat = [1, 1]\n[[teleporter]]\nat_mm = [50.5, 60]\n[[zone]]\nname = "grey"\nfrom = [0, 0]\nto = [7, 1]\n', cfg)
    assert m.objectives[0] == pytest.approx(mm(1, 1)) and m.teleporters[0] == (50.5, 60.0)
    assert m.zones[0].rect == pytest.approx((0.0, 0.0, cfg.width_mm, mm(0, 1)[1]))


def test_shape_sizes():
    cfg = tomb_world_grid()
    shapes = ter.shapes(parse(wall_map((0, 1), (1, 1), [(0, 1), (1, 1)], True), cfg), cfg)
    wall, hatch = by_kind(shapes, "wall")[0], by_kind(shapes, "hatchway")[0]
    assert size_along(wall, 1) == pytest.approx(4.0) and size_along(wall, 0) == pytest.approx(PITCH)
    assert (size_along(hatch, 0), size_along(hatch, 1)) == pytest.approx((22.0, 10.0))
    assert hatch.points.mean(axis=0) == pytest.approx(np.add(mm(0, 1), (PITCH / 2, 0)))
    assert all(size_along(p, 0) == pytest.approx(13.0) for p in by_kind(shapes, "pillar"))


def test_breach_width_by_wall_length():
    cfg = tomb_world_grid()
    short = ter.shapes(parse(wall_map((3, 3), (3, 4), [(3, 3), (3, 4)], True, key="B2"), cfg), cfg)
    long = ter.shapes(parse(wall_map((3, 1), (5, 1), [(3, 1), (4, 1), (5, 1)], True, key="A1", at=(4.5, 1)), cfg), cfg)
    b = by_kind(short, "breach")[0]
    assert (size_along(b, 1), size_along(b, 0)) == pytest.approx((28.0, 4.0))
    assert size_along(by_kind(long, "breach")[0], 0) == pytest.approx(35.0)


def test_opening_between_pillars_of_an_a_wall():
    cfg = tomb_world_grid()
    pillars = [(3, 1), (4, 1), (5, 1)]
    m = parse(wall_map((3, 1), (5, 1), pillars, True, key="A1", at=(4.5, 1)), cfg)
    assert m.walls[0].opening_at == pytest.approx(mm(4.5, 1)) and m.walls[0].opening == "breach"
    with pytest.raises(ter.MapError, match="give opening_at"):
        parse(wall_map((3, 1), (5, 1), pillars, True, key="A1"), cfg)
    with pytest.raises(ter.MapError, match="centre between two pillars"):
        parse(wall_map((3, 1), (5, 1), pillars, True, key="A1", at=(4, 1)), cfg)
    a2 = parse(wall_map((3, 1), (5, 1), [(3, 1), (5, 1)], True, key="A2"), cfg)  # no middle pillar: wall middle
    assert a2.walls[0].opening_at == pytest.approx(mm(4, 1)) and a2.walls[0].opening == "hatchway"


def test_wall_key_gives_opening_kind_and_middle_pillar():
    cfg = tomb_world_grid()
    assert parse(wall_map((3, 3), (3, 4), [(3, 3), (3, 4)], True, key="B2"), cfg).walls[0].opening == "breach"
    assert parse(wall_map((3, 3), (3, 4), [(3, 3), (3, 4)], True, key="B1"), cfg).walls[0].opening == "hatchway"
    assert parse(wall_map((3, 3), (3, 4), [(3, 3), (3, 4)], False, key="B2"), cfg).walls[0].opening == "none"
    with pytest.raises(ter.MapError, match="add it to pillars"):
        parse(wall_map((3, 1), (5, 1), [(3, 1), (5, 1)], key="A4"), cfg)
    with pytest.raises(ter.MapError, match="remove it from pillars"):
        parse(wall_map((3, 1), (5, 1), [(3, 1), (4, 1), (5, 1)], key="A2"), cfg)


def test_wall_keys_above_horizontal_and_right_of_vertical_walls():
    cfg = tomb_world_grid()
    text = wall_map((0, 1), (1, 1), [(0, 1), (1, 1), (0, 2)], key="B1") + '[[wall]]\nkey = "B4"\nfrom = [0, 1]\nto = [0, 2]\n'
    keys = {s.name: s for s in by_kind(ter.shapes(parse(text, cfg), cfg), "wall_key")}
    flat, upright = keys["B1"], keys["B4"]
    assert flat.center == pytest.approx(mm(0.5, 1)) and flat.points[0, 1] < flat.center[1] and not flat.vertical
    assert upright.center == pytest.approx(mm(0, 1.5)) and upright.points[0, 0] > upright.center[0] and upright.vertical


def test_wall_end_is_half_a_pillar_past_the_wall_end():
    cfg = tomb_world_grid()
    text = wall_map((5, 3), (6, 3), [(5, 3)]).replace("pillars", "wall_ends = [[6, 3]]\npillars")
    shapes = ter.shapes(parse(text, cfg), cfg)
    end = by_kind(shapes, "wall_end")[0]
    x6 = mm(6, 3)[0]
    assert (size_along(end, 0), size_along(end, 1)) == pytest.approx((6.5, 13.0))
    assert end.points[:, 0].min() == pytest.approx(x6) and end.points[:, 0].max() == pytest.approx(x6 + 6.5)


@pytest.mark.parametrize(
    "text, message",
    [
        (wall_map((0, 1), (1, 2), [(0, 1), (1, 2)]), "horizontal or vertical"),
        (wall_map((0, 1), (3, 1), [(0, 1), (3, 1)], key="A1"), "2 grid long"),
        (wall_map((0, 1), (2, 1), [(0, 1), (1, 1), (2, 1)], key="B1"), "1 grid long"),
        (wall_map((0, 1), (1, 1), [(0, 1), (1, 1)], key="C1"), "key must be"),
        (wall_map((0, 1), (1, 1), [(0, 1), (1, 1)]).replace("opening = false", 'opening = "hatchway"'), "opening must be true"),
        (wall_map((0, 1), (1, 1), [(0, 1), (1, 1)], at=(0.5, 1)), "needs opening = true"),
        (wall_map((0, 1.25), (1, 1.25), [(0, 1.25), (1, 1.25)]), "half grid"),
        (wall_map((0, 1), (1, 1), [(0, 1)]), "no pillar"),
        (wall_map((0, 1), (1, 1), [(0, 1), (1, 1), (1, 1)]), "two pillars"),
        (wall_map((0, 1), (1, 1), [(0, 1), (1, 1)], extra="wall_ends = [[3, 3]]"), "touch exactly 1 wall"),
        (wall_map((0, 1), (9, 1), [(0, 1), (9, 1)]), "outside the grid"),
        (HEAD.replace("schema = 1", "schema = 2"), "schema"),
        (HEAD + "walls = []\n", "unknown keys"),
        (HEAD + "[[objective]]\nat = [1, 1]\nat_mm = [1, 1]\n", "either at"),
    ],
)
def test_validation_errors(text, message):
    with pytest.raises(ter.MapError, match=message):
        parse(text)


def test_list_maps_groups_by_terrain_and_reports_bad_files(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.toml").write_text(HEAD.replace('"T"', '"Two"'))
    (tmp_path / "a" / "bad.toml").write_text("not toml [")
    entries, problems = ter.list_maps(tmp_path)
    assert [(e.terrain, e.name) for e in entries] == [("Tomb World", "Two")] and len(problems) == 1


def test_draw_terrain_marks_the_wall(tmp_path):
    cfg = tomb_world_grid()
    img, mm2px = render_board(cfg)
    before = img.copy()
    shapes = ter.shapes(parse(wall_map((0, 1), (2, 1), [(0, 1), (2, 1)], key="A2"), cfg), cfg)
    to_px = lambda pts: np.array([mm2px(p) for p in np.asarray(pts).reshape(-1, 2)])  # noqa: E731
    draw_terrain(img, shapes, to_px)
    x, y = np.rint(mm2px(np.array(mm(1, 1)))).astype(int)
    assert not np.array_equal(img[y, x], before[y, x])


def test_map_command_exit_codes(tmp_path, capsys):
    assert main(["map", str(CARD_6), "--out", str(tmp_path)]) == 0
    assert (tmp_path / "layout-6.png").exists()
    bad = tmp_path / "bad.toml"
    bad.write_text(wall_map((0, 1), (1, 2), [(0, 1), (1, 2)]))
    assert main(["map", str(bad)]) == 1
    assert main(["map", str(tmp_path / "missing.toml")]) == 2


def test_dashed_circle_one_inch_around_each_objective():
    cfg = tomb_world_grid()
    shapes = ter.shapes(parse(HEAD + "[[objective]]\nat = [1, 1]\n", cfg), cfg)
    ring, marker = by_kind(shapes, "objective_range")[0], by_kind(shapes, "objective")[0]
    assert marker.radius == pytest.approx(20.0) and ring.radius == pytest.approx(20.0 + 25.4)
    assert ring.center == marker.center


def test_keys_come_last_with_a_t_on_each_teleporter():
    cfg = tomb_world_grid()
    m = ter.load_map(CARD_6, cfg)
    kinds = [s.kind for s in ter.shapes(m, cfg)]
    first_key = min(kinds.index("wall_key"), kinds.index("teleporter_key"), kinds.index("light_key"))
    assert set(kinds[first_key:]) == {"wall_key", "teleporter_key", "light_key"}
    pads = [s for s in ter.shapes(m, cfg) if s.kind == "teleporter_key"]
    assert [p.name for p in pads] == ["T", "T"] and pads[0].center == pytest.approx(m.teleporters[0])


@pytest.mark.parametrize("key", ["B2", "A2"])  # B2: breach point (as thin as the wall), A2: hatchway
def test_open_opening_is_a_gap_in_the_wall(key):
    from dataclasses import replace

    cfg = tomb_world_grid()
    floor = np.full((800, 800, 3), 90, np.uint8)
    end = (1, 1) if key == "B2" else (2, 1)
    shapes = ter.shapes(parse(wall_map((0, 1), end, [(0, 1), end], opening=True, key=key), cfg), cfg)
    opening = next(sh for sh in shapes if sh.kind in ("hatchway", "breach"))
    assert opening.kind == ("breach" if key == "B2" else "hatchway")
    to_px = lambda pts: np.asarray(pts, float).reshape(-1, 2) * 2.0  # noqa: E731
    centre = np.rint(np.mean(opening.points, axis=0) * 2.0).astype(int)
    closed = floor.copy()
    draw_terrain(closed, shapes, to_px)
    opened = floor.copy()
    draw_terrain(opened, [replace(sh, open=True) if sh is opening else sh for sh in shapes], to_px)
    x, y = centre
    assert not np.array_equal(closed[y, x], floor[y, x])  # filled
    assert np.array_equal(opened[y, x], floor[y, x])  # the floor shows through
    assert not np.array_equal(opened, closed)


def test_light_terrain_footprints():
    cfg = tomb_world_grid()
    text = HEAD + '[[light]]\nkey = "C2"\nat = [1, 1]\nlong = "y"\n[[light]]\nkey = "C1"\nat_mm = [200, 300]\n[[light]]\nkey = "C5"\nat = [3, 3]\n'
    m = parse(text, cfg)
    assert [(i.key, i.long_axis) for i in m.light] == [("C2", "y"), ("C1", "x"), ("C5", "x")]
    light = by_kind(ter.shapes(m, cfg), "light")
    c2, c1, c5 = light
    assert (size_along(c2, 0), size_along(c2, 1)) == pytest.approx((46.0, 62.0)) and c2.center == pytest.approx(mm(1, 1))
    assert len(c1.points) == 8 and (size_along(c1, 0), size_along(c1, 1)) == pytest.approx((81.0, 63.0))
    top = c1.points[np.isclose(c1.points[:, 1], 300 - 31.5)]
    assert float(np.ptp(top[:, 0])) == pytest.approx(81.0 - 2 * 16.0)  # corner cut
    assert len(c5.points) == 4 and size_along(c5, 0) == pytest.approx(40.0)
    keys = by_kind(ter.shapes(m, cfg), "light_key")
    assert [k.name for k in keys] == ["C2", "C1", "C5"]


def test_light_terrain_is_on_the_floor_and_does_not_block():
    from battlesight import walls

    cfg = tomb_world_grid()
    m = parse(HEAD + '[[light]]\nkey = "C3"\nat = [3, 3]\n', cfg)
    shapes = ter.shapes(m, cfg)
    assert [s.kind for s in shapes] == ["light", "light_key"]
    assert len(walls.obstacles(shapes)) == 0


@pytest.mark.parametrize(
    "entry, message",
    [
        ('key = "C9"\nat = [3, 3]', "light terrain label"),
        ('key = "A1"\nat = [3, 3]', "light terrain label"),
        ('key = "C1"\nat = [3, 3]\nlong = "z"', 'long must be "x" or "y"'),
        ('key = "C1"\nat = [0, 3]', "over the board edge"),
        ('key = "C1"\nat = [3, 3]\nturn = 90', "unknown keys turn"),
    ],
)
def test_light_terrain_errors(entry, message):
    with pytest.raises(ter.MapError, match=message):
        parse(HEAD + "[[light]]\n" + entry + "\n")


def test_light_terrain_may_touch_the_board_edge():
    cfg = tomb_world_grid()
    m = parse(HEAD + '[[light]]\nkey = "C1"\nat_mm = [662.5, 400]\n', cfg)  # right edge at 703 mm
    assert m.light[0].at == (662.5, 400.0)


def test_light_sizes_from_the_board_file(tmp_path):
    from battlesight.config import DEFAULT_CONFIG, BoardConfig

    cfg = BoardConfig.load(DEFAULT_CONFIG)
    assert cfg.light_size("C1") == (81.0, 63.0, 16.0) and cfg.light_size("C2") == (62.0, 46.0, 0.0)
    assert cfg.light_size("C9") is None
    text = open(DEFAULT_CONFIG).read()
    bad = tmp_path / "board.toml"
    bad.write_text(text.replace("C2 = [62.0, 46.0]", "C2 = [46.0, 62.0]"))
    with pytest.raises(ValueError, match="light_mm.C2"):
        BoardConfig.load(bad)


# --- piece files ------------------------------------------------------------------

L_PIECE = [[0, 0], [60, 0], [60, 20], [20, 20], [20, 40], [0, 40]]  # an L, bounding box 60 x 40


def piece_text(points=L_PIECE, key="C2", extra="") -> str:
    return f'schema = 1\nkey = "{key}"\noutline_mm = {points}\n{extra}'


def test_piece_file_gives_the_real_outline(tmp_path):
    import tomllib

    piece = ter.parse_piece(tomllib.loads(piece_text()))
    assert piece.key == "C2"
    assert piece.outline.min(axis=0) == pytest.approx((-30.0, -20.0)) and piece.outline.max(axis=0) == pytest.approx((30.0, 20.0))
    cfg = tomb_world_grid()
    m = parse(HEAD + '[[light]]\nkey = "C2"\nat_mm = [200, 300]\nlong = "y"\n[[light]]\nkey = "C5"\nat = [3, 3]\n', cfg)
    c2, c5 = by_kind(ter.shapes(m, cfg, {"C2": piece}), "light")
    assert len(c2.points) == 4 and not c2.located and c2.center == (200.0, 300.0)  # not found on the board: the card symbol
    assert (size_along(c2, 0), size_along(c2, 1)) == pytest.approx((46.0, 62.0))  # long = "y"
    assert len(c5.points) == 4  # no piece file: the card symbol
    found = ter.shapes(m, cfg, {"C2": piece}, {0: ((250.0, 310.0), 0.0), 1: ((0.0, 0.0), 45.0)})
    c2, c5 = by_kind(found, "light")
    assert len(c2.points) == 6 and c2.located and not c5.located
    assert (size_along(c2, 0), size_along(c2, 1)) == pytest.approx((60.0, 40.0)) and c2.center == (250.0, 310.0)
    assert c2.points.min(axis=0) == pytest.approx((220.0, 290.0))
    assert c5.center == pytest.approx(mm(3, 3))  # a pose without a piece file is not used
    assert [k.center for k in by_kind(found, "light_key")] == [c2.center, c5.center]


def test_piece_files_of_a_map_folder(tmp_path):
    cfg = tomb_world_grid()
    maps, root = tmp_path / "maps" / "tomb-world", tmp_path / "terrain"
    maps.mkdir(parents=True)
    (maps / "a.toml").write_text(HEAD)
    tmap = ter.load_map(maps / "a.toml", cfg)
    assert ter.pieces_for(tmap, root) == ({}, [])  # no folder
    folder = root / "tomb-world"
    folder.mkdir(parents=True)
    piece = ter.Piece("C3", ter.piece_frame(np.array(L_PIECE, float)))
    ter.write_piece(folder / "C3.toml", piece, source="photo.jpg")
    (folder / "bad.toml").write_text(piece_text(points=[[0, 0], [1, 1]]))
    pieces, problems = ter.pieces_for(tmap, root)
    assert list(pieces) == ["C3"]
    assert np.allclose(pieces["C3"].outline, piece.outline, atol=0.05)
    assert len(problems) == 1 and "bad.toml" in problems[0] and "3 or more points" in problems[0]


@pytest.mark.parametrize(
    "text, message",
    [
        (piece_text().replace("schema = 1", "schema = 2"), "schema"),
        (piece_text(key=""), "key"),
        (piece_text(extra="height_mm = 18"), "unknown keys height_mm"),
        (piece_text(points=[[0, 0], [10, 0], [20, 0]]), "no area"),
        (piece_text(extra="colour = 1"), "unknown keys colour"),
    ],
)
def test_piece_file_errors(text, message):
    import tomllib

    with pytest.raises(ter.MapError, match=message):
        ter.parse_piece(tomllib.loads(text))


# --- skill record-terrain: outline.py ---------------------------------------------


def outline_tool():
    import importlib.util
    import sys
    from pathlib import Path

    if "record_terrain_outline" not in sys.modules:
        path = Path(__file__).resolve().parents[1] / ".claude" / "skills" / "record-terrain" / "outline.py"
        spec = importlib.util.spec_from_file_location("record_terrain_outline", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["record_terrain_outline"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["record_terrain_outline"]


def underside_photo(angle_deg=8.0, ppm=6.0, at=(700.0, 500.0)):
    """A light table with the dark underside of the L piece, turned.
    Returns the photo and the outline in photo pixels."""
    import cv2

    a = np.radians(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    px = ((np.array(L_PIECE, float) - (30.0, 20.0)) @ rot.T) * ppm + at
    img = np.full((1000, 1400, 3), 225, np.uint8)
    cv2.fillPoly(img, [np.rint(px).astype(np.int32)], (60, 60, 60), cv2.LINE_AA)
    return img, px


def test_underside_outline_is_scaled_and_mirrored():
    tool = outline_tool()
    _, px = underside_photo()
    outline, ratio, off, spread = tool.to_piece(px, 60.0, 40.0)
    assert ratio < 0.004 and off == pytest.approx(8.0, abs=1.0) and spread < 0.02  # the frame follows the sizes, not the photo
    assert np.ptp(outline, axis=0) == pytest.approx((60.0, 40.0), abs=0.3)
    assert outline.min(axis=0) == pytest.approx(-outline.max(axis=0))
    # One long slanted edge must not turn the frame (the minimum area rectangle does).
    cut = np.array([[0, 0], [460, 0], [460, 300], [200, 620], [0, 620]], float)
    assert tool.to_piece(cut, 62.0, 46.0)[2] == pytest.approx(0.0, abs=0.5)
    # The L seen from above is the mirror image: the turn direction of the points changes.
    def turn(p):
        return np.sign(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1)))

    assert turn(outline) == -turn(px)
    # Same shape as the true L or the L turned by 180 degrees, mirrored in x.
    truth = (np.array(L_PIECE, float) - (30.0, 20.0)) * (-1.0, 1.0)
    assert any(np.allclose(np.sort(outline, axis=0), np.sort(t, axis=0), atol=1.0) for t in (truth, -truth))
    # Sizes that do not agree with the outline show in the ratio.
    assert tool.to_piece(px, 60.0, 30.0)[1] > tool.MAX_RATIO  # no frame angle within 20 degrees agrees
    with pytest.raises(tool.OutlineError, match="longer"):
        tool.to_piece(px, 40.0, 60.0)


def test_underside_suggest_and_view():
    import cv2

    tool = outline_tool()
    photo, px = underside_photo()
    found = tool.suggest(photo)
    far = max(abs(cv2.pointPolygonTest(px.astype(np.float32), (float(x), float(y)), True)) for x, y in found)
    assert far < 4.0  # px: within 0.7 mm of the true edge
    with pytest.raises(tool.OutlineError, match="no piece"):
        tool.suggest(np.full((400, 400, 3), 225, np.uint8))
    img = tool.view(photo, found, box=(400, 250, 1000, 750))
    assert max(img.shape[:2]) == tool.VIEW_PX
    assert (img != tool.view(photo, box=(400, 250, 1000, 750))).any()  # the outline is drawn
    with pytest.raises(tool.OutlineError, match="empty"):
        tool.view(photo, box=(10, 10, 10, 10))


def test_underside_command_writes_a_piece_file(tmp_path, capsys):
    import json

    import cv2

    tool = outline_tool()
    photo, px = underside_photo()
    cv2.imwrite(str(tmp_path / "c2.png"), photo)
    points = tmp_path / "points.json"
    assert tool.main(["suggest", str(tmp_path / "c2.png"), "--out", str(points)]) == 0
    assert tool.main(["view", str(tmp_path / "c2.png"), "--points", str(points), "--out", str(tmp_path / "view.png")]) == 0
    before = json.loads(points.read_text())
    assert tool.main(["simplify", str(points), "--length", "60", "--tolerance-mm", "1", "--out", str(points)]) == 0
    after = json.loads(points.read_text())
    assert 3 <= len(after) < len(before) and all(p in before for p in after)  # a subset, in the same order
    save = ["save", str(points), "--key", "C2", "--terrain", "test-world", "--root", str(tmp_path / "terrain"), "--check", str(tmp_path / "check.png")]
    assert tool.main([*save, "--length", "60", "--width", "40", "--source", "c2.png"]) == 0
    assert "C2.toml" in capsys.readouterr().out and (tmp_path / "check.png").exists()
    pieces, problems = ter.load_pieces(tmp_path / "terrain" / "test-world")
    assert problems == []
    assert np.ptp(pieces["C2"].outline, axis=0) == pytest.approx((60.0, 40.0), abs=0.2)
    assert tool.main([*save, "--length", "60", "--width", "30"]) == 1  # the sizes do not agree with the outline
    points.write_text(json.dumps([[0, 0], [1, 1]]))
    assert tool.main([*save, "--length", "60", "--width", "40"]) == 2
    assert tool.main(["simplify", str(points), "--length", "60", "--out", str(points)]) == 2
    assert tool.main(["view", str(tmp_path / "none.png"), "--out", str(tmp_path / "v.png")]) == 2
