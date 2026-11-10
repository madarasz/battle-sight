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
    first_key = min(kinds.index("wall_key"), kinds.index("teleporter_key"))
    assert set(kinds[first_key:]) == {"wall_key", "teleporter_key"}
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
