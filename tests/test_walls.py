"""Measurement around walls: shortest path and range area."""

from __future__ import annotations

import numpy as np
import pytest

from battlesight.measure import MM_PER_INCH, rim_gap, ring_radius_mm
from battlesight.terrain import Shape
from battlesight.walls import Navigator, contains, obstacles

WALL = (98.0, 50.0, 102.0, 150.0)  # vertical, between the two bases
A, B, D = (60.0, 100.0), (140.0, 100.0), 32.0
TO_CORNER = float(np.hypot(38.0, 50.0)) - 16.0  # rim of A to the wall end corner (98, 50)
AROUND = 2 * TO_CORNER + 4.0  # rim, corner, corner, rim
RANGE = 3 * MM_PER_INCH


def rect(x0, y0, x1, y1) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def hatch_wall(is_open: bool) -> list[Shape]:
    return [Shape("wall", rect(0, 98, 200, 102)), Shape("hatchway", rect(89, 95, 111, 105), open=is_open)]


def test_no_obstacle_gives_the_straight_rim_gap():
    length, points = Navigator([]).path(A, D, B, 40.0)
    gap, start, end = rim_gap(A, D, B, 40.0)
    assert length == pytest.approx(gap)
    assert points == pytest.approx(np.array([start, end]))


def test_path_goes_around_the_wall_end():
    length, points = Navigator([WALL]).path(A, D, B, D)
    assert length == pytest.approx(AROUND, abs=0.5)
    assert len(points) == 4
    assert {tuple(p) for p in points[1:3]} in ({(98.0, 50.0), (102.0, 50.0)}, {(98.0, 150.0), (102.0, 150.0)})
    assert np.linalg.norm(np.diff(points, axis=0), axis=1).sum() == pytest.approx(length)
    assert np.linalg.norm(points[0] - A) == pytest.approx(16.0) and np.linalg.norm(points[-1] - B) == pytest.approx(16.0)


def test_path_is_the_same_in_both_directions():
    nav = Navigator([WALL, (60.0, 20.0, 73.0, 33.0)])
    there = nav.path((50.0, 80.0), 25.0, (150.0, 120.0), 40.0)
    back = nav.path((150.0, 120.0), 40.0, (50.0, 80.0), 25.0)
    assert there[0] == pytest.approx(back[0], abs=0.3)


def test_line_along_a_wall_face_or_through_a_corner_is_clear():
    nav = Navigator([WALL])
    assert nav.clear((98.0, 0.0), [(98.0, 200.0)])[0]  # along the face
    assert nav.clear((88.0, 60.0), [(108.0, 40.0)])[0]  # touches the corner (98, 50)
    assert not nav.clear((90.0, 42.0), [(106.0, 58.0)])[0]  # cuts the corner
    assert not nav.clear(A, [B])[0]


def test_closed_hatchway_blocks_and_open_hatchway_is_a_gap():
    closed, opened = obstacles(hatch_wall(False)), obstacles(hatch_wall(True))
    assert sorted(map(tuple, closed)) == [(0, 98, 200, 102), (89, 95, 111, 105)]
    assert sorted(map(tuple, opened)) == [(0, 98, 89, 102), (111, 98, 200, 102)]
    up, down = (100.0, 60.0), (100.0, 140.0)
    straight = rim_gap(up, D, down, D)[0]
    assert Navigator(closed).path(up, D, down, D)[0] > straight + 100.0  # around a wall end
    assert Navigator(opened).path(up, D, down, D)[0] == pytest.approx(straight)


def test_obstacles_take_pillars_and_wall_ends_but_no_zones_or_discs():
    shapes = [Shape("zone", rect(0, 0, 300, 50)), Shape("pillar", rect(10, 10, 23, 23)), Shape("wall_end", rect(40, 10, 46.5, 23)), Shape("teleporter", rect(100, 100, 160, 160))]
    assert sorted(map(tuple, obstacles(shapes))) == [(10, 10, 23, 23), (40, 10, 46.5, 23)]


def test_closed_room_has_no_path_and_keeps_the_range_inside():
    room = [(0, 0, 100, 4), (0, 96, 100, 100), (0, 0, 4, 100), (96, 0, 100, 100)]
    nav = Navigator(room)
    assert nav.path((50.0, 50.0), D, (200.0, 50.0), D) is None
    area = nav.reach((50.0, 50.0), D, 6 * MM_PER_INCH)
    assert contains(area, (10.0, 10.0)) and not contains(area, (120.0, 50.0))
    assert np.isinf(nav.distance_to((50.0, 50.0), D, (120.0, 50.0)))


def test_range_without_obstacles_is_the_ring():
    center = np.array([300.0, 300.0])
    (outline,) = Navigator([]).reach(center, D, RANGE)
    radius = np.linalg.norm(outline - center, axis=1)
    assert np.abs(radius - ring_radius_mm(D, 3)).max() < 0.5


def test_range_stops_at_a_wall_and_bends_around_its_end():
    nav = Navigator([WALL], size_mm=(703.0, 606.0))
    area = nav.reach(A, D, RANGE)
    behind, past_end = (130.0, 100.0), (110.0, 45.0)
    assert np.linalg.norm(np.subtract(behind, A)) - 16.0 < RANGE  # in the plain ring
    assert nav.distance_to(A, D, behind) == pytest.approx(TO_CORNER + 4.0 + np.hypot(28.0, 50.0), abs=0.5)
    assert not contains(area, behind)
    assert nav.distance_to(A, D, past_end) == pytest.approx(TO_CORNER + 13.0, abs=0.5)
    assert contains(area, past_end)
    assert contains(area, (30.0, 100.0)) and not contains(area, (100.0, 100.0))  # open floor, in the wall


def test_range_edge_is_at_the_range_from_the_rim():
    nav = Navigator([WALL, (60.0, 20.0, 73.0, 33.0)], size_mm=(703.0, 606.0))
    outlines = nav.reach(A, D, RANGE)
    outer = max(outlines, key=len)
    free = outer[(outer[:, 0] > 1.0) & (outer[:, 1] > 1.0)]  # not the board edge
    dist = np.array([nav.distance_to(A, D, p) for p in free[::25]])
    on_wall = dist < RANGE - 1.0  # the edge along a wall face is nearer than the range
    assert np.abs(dist[~on_wall] - RANGE).max() < 1.0
    assert on_wall.mean() < 0.5


def test_pillar_in_the_range_is_a_hole():
    pillar = (80.0, 94.0, 93.0, 107.0)
    nav = Navigator([pillar])
    area = nav.reach(A, D, RANGE)
    assert len(area) == 2
    assert not contains(area, (86.5, 100.5)) and contains(area, (110.0, 100.0))


def test_range_is_cut_at_the_board_edge():
    area = Navigator([], size_mm=(703.0, 606.0)).reach((20.0, 20.0), D, RANGE)
    points = np.concatenate(area)
    assert points.min() >= 0.0 and not contains(area, (-5.0, 20.0))


def test_range_does_not_leak_behind_a_long_wall_next_to_the_base():
    # The base is near a long wall: the shadow of the wall is almost a half plane.
    nav = Navigator([(0.0, 96.0, 400.0, 100.0)], size_mm=(703.0, 606.0))
    center = (200.0, 120.0)
    area = nav.reach(center, D, 6 * MM_PER_INCH)
    assert len(area) == 1
    for x in (60.0, 120.0, 200.0, 280.0, 340.0):
        for y in (20.0, 60.0, 90.0):
            assert not contains(area, (x, y)), (x, y)
    assert contains(area, (200.0, 250.0)) and contains(area, (70.0, 110.0))


def test_pillar_on_the_outer_grid_line_blocks_as_far_as_the_board_edge():
    size, outer = (703.0, 606.0), (12.0, 12.0, 691.0, 594.0)
    shapes = [Shape("pillar", rect(5.5, 296.5, 18.5, 309.5)), Shape("wall", rect(12, 301, 109, 305)), Shape("pillar", rect(102.5, 296.5, 115.5, 309.5)), Shape("pillar", rect(300, 300, 313, 313))]
    plain, edge = obstacles(shapes), obstacles(shapes, size, outer)
    assert (5.5, 296.5, 18.5, 309.5) in set(map(tuple, plain))
    assert sorted(map(tuple, edge)) == [(-1.0, 296.5, 18.5, 309.5), (-1.0, 301, 109, 305), (102.5, 296.5, 115.5, 309.5), (300, 300, 313, 313)]
    up, down = (40.0, 270.0), (40.0, 340.0)  # each side of the wall, near the board edge
    through_margin = Navigator(plain, size).path(up, D, down, D)
    around = Navigator(edge, size).path(up, D, down, D)
    assert through_margin[1][:, 0].min() < 6.0  # the bug: around the pillar, in the printed margin
    assert around[1][:, 0].max() > 115.0 and around[0] > through_margin[0] + 50.0  # around the other wall end
    area = Navigator(edge, size).reach(up, D, 3 * MM_PER_INCH)
    assert contains(area, (3.0, 280.0)) and not contains(area, (3.0, 320.0))  # the margin on the far side of the pillar


def test_terrain_on_each_outer_grid_line_goes_to_its_edge():
    size, outer = (703.0, 606.0), (12.0, 12.0, 691.0, 594.0)
    shapes = [Shape("pillar", rect(684.5, 587.5, 697.5, 600.5)), Shape("pillar", rect(200, 5.5, 213, 18.5))]
    assert sorted(map(tuple, obstacles(shapes, size, outer))) == [(200, -1.0, 213, 18.5), (684.5, 587.5, 704.0, 607.0)]
