"""Movement of a base: steps of 1 in around terrain."""

from __future__ import annotations

import numpy as np
import pytest

from battlesight.measure import MM_PER_INCH
from battlesight.moves import move_field
from battlesight.terrain import Shape
from battlesight.walls import contains, obstacles, open_gaps

START, D = (300.0, 300.0), 32.0


def seg_dist(points: np.ndarray, at) -> float:
    """Shortest distance from a chain of segments to a point."""
    at, best = np.asarray(at, float), np.inf
    for a, b in zip(points[:-1], points[1:]):
        ab = b - a
        t = np.clip(np.dot(at - a, ab) / max(float(np.dot(ab, ab)), 1e-9), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(a + t * ab - at)))
    return best


@pytest.mark.parametrize("dist_mm, inches", [(20.0, 1), (30.0, 2), (60.0, 3), (69.85, 3), (100.0, 4), (140.0, 6)])
def test_open_floor_costs_the_length_rounded_up(dist_mm, inches):
    field = move_field(START, D, 6)
    for angle in (0.0, 0.7, 2.0, 4.1):
        at = np.asarray(START) + dist_mm * np.array([np.cos(angle), np.sin(angle)])
        assert field.cost(at) == inches
        total, points, costs = field.route(at)
        assert total == inches and costs == [inches] and len(points) == 2  # one straight increment
        assert points[0] == pytest.approx(START) and points[-1] == pytest.approx(at)


def test_budget_limits_the_reach_and_the_outline_is_the_ring():
    field = move_field(START, D, 3)
    assert field.cost((300.0 + 3 * MM_PER_INCH - 2.0, 300.0)) == 3
    assert field.cost((300.0 + 3 * MM_PER_INCH + 2.0, 300.0)) is None
    (outline,) = field.outline()
    radius = np.linalg.norm(outline - START, axis=1)
    assert np.abs(radius - (16.0 + 3 * MM_PER_INCH)).max() < 1.5


def test_the_base_goes_around_a_wall_with_its_full_width():
    wall = (296.0, 250.0, 300.0, 350.0)  # vertical
    start, goal = (250.0, 300.0), (350.0, 300.0)
    field = move_field(start, D, 9, [wall])
    total, points, costs = field.route(goal)
    assert total > 4 and sum(costs) == total and len(points) >= 3  # 100 mm straight is 4 in
    for corner in ((296.0, 250.0), (300.0, 250.0), (296.0, 350.0), (300.0, 350.0)):
        assert seg_dist(points, corner) > 16.0 - 2.0  # the rim stays out of the wall
    assert field.cost((283.0, 300.0)) is None and field.cost((278.0, 300.0)) is not None  # centre to wall: 16 mm


def test_a_hole_in_a_wall_must_be_as_wide_as_the_base():
    rects = [(0.0, 298.0, 285.0, 302.0), (315.0, 298.0, 600.0, 302.0)]  # hole of 30 mm, no hatchway
    assert move_field((300.0, 260.0), 32.0, 6, rects).cost((300.0, 340.0)) is None
    assert move_field((300.0, 260.0), 25.0, 6, rects).cost((300.0, 340.0)) == 4  # 80 mm


def test_open_hatchway_lets_each_base_through_for_one_inch_more():
    shapes = [Shape("wall", np.array([[0, 298], [600, 298], [600, 302], [0, 302]], float)), Shape("hatchway", np.array([[289, 295], [311, 295], [311, 305], [289, 305]], float), open=True)]
    rects, gaps = obstacles(shapes), open_gaps(shapes)
    assert gaps.tolist() == [[289, 295, 311, 305]]
    field = move_field((300.0, 260.0), 40.0, 6, rects, gaps)  # the base is wider than the gap
    total, points, costs = field.route((300.0, 340.0))
    assert total == 5 and costs == [5] and len(points) == 2  # 80 mm = 4 in, and 1 in for the hatchway
    assert field.cost((300.0, 272.0)) == 1 and field.cost((300.0, 330.0)) == 4  # 70 mm = 3 in, and 1
    assert field.cost((300.0, 288.0)) is None and field.cost((300.0, 312.0)) is None  # no stop with the base over the wall
    assert field.cost((340.0, 285.0)) is None  # beside the gap the base keeps away from the wall
    total, points, costs = field.route((318.0, 350.0))  # a slanted line through the gap: 92 mm
    assert (total, costs, len(points)) == (5, [5], 2)
    bent = move_field((240.0, 260.0), 40.0, 9, rects, gaps).route((240.0, 340.0))  # the gap is not on the line
    assert bent[0] == sum(bent[2]) and len(bent[1]) >= 3 and 289.0 < np.interp(300.0, bent[1][:, 1], bent[1][:, 0]) < 311.0
    closed = move_field((300.0, 260.0), 40.0, 6, obstacles([shapes[0]]))
    assert closed.cost((300.0, 340.0)) is None


def test_enemy_base_blocks_the_way():
    enemy = ((300.0, 300.0), 40.0)
    start, goal = (230.0, 300.0), (370.0, 300.0)  # 140 mm: 6 in
    assert move_field(start, D, 9).route(goal)[0] == 6
    field = move_field(start, D, 9, enemies=[enemy])
    total, points, costs = field.route(goal)
    assert total >= 6 and len(points) >= 3 and sum(costs) == total
    assert seg_dist(points, enemy[0]) > 16.0 + 20.0 - 2.0  # the rims do not overlap
    assert field.cost((300.0, 300.0)) is None and field.cost((266.0, 300.0)) is None and field.cost((262.0, 300.0)) == 2


def test_a_base_that_touches_a_wall_can_move_away():
    wall = (0.0, 316.0, 600.0, 320.0)
    field = move_field((300.0, 301.5), D, 3, [wall])  # the rim is 1.5 mm in the wall
    assert field.cost((300.0, 270.0)) == 2 and field.cost((340.0, 299.0)) == 2
    assert field.cost((300.0, 340.0)) is None


def test_the_base_stays_on_the_board():
    field = move_field((40.0, 40.0), D, 3, size_mm=(703.0, 606.0))
    assert field.cost((17.0, 40.0)) == 1 and field.cost((12.0, 40.0)) is None
    points = np.concatenate(field.outline())
    assert points.min() > -1.5


def test_point_on_the_covered_area_goes_to_the_nearest_centre_in_reach():
    field = move_field(START, D, 2)
    limit = 2 * MM_PER_INCH
    assert contains(field.outline(), (300.0 + limit + 10.0, 300.0))
    total, points, _ = field.route((300.0 + limit + 10.0, 300.0))  # the centre cannot get here, the rim can
    assert total == 2 and points[-1] == pytest.approx((300.0 + limit, 300.0), abs=1.5)
    assert field.route((300.0 + limit + 16.0 + 6.0, 300.0)) is None


def test_the_base_goes_through_a_friend_but_cannot_stop_on_it():
    friend = ((300.0, 300.0), 40.0)
    start, goal = (230.0, 300.0), (370.0, 300.0)  # 140 mm: 6 in
    field = move_field(start, D, 6, friends=[friend])
    total, points, costs = field.route(goal)
    assert (total, costs, len(points)) == (6, [6], 2)  # straight through the friend
    assert field.cost((300.0, 300.0)) is None and field.cost((268.0, 300.0)) is None  # the bases would overlap
    assert field.cost((262.0, 300.0)) == 2 and field.cost((338.0, 300.0)) == 5
    area = field.outline()
    assert len(area) == 2  # the friend is a hole in the area
    assert not contains(area, (300.0, 300.0)) and not contains(area, (300.0, 315.0))
    assert contains(area, (300.0, 325.0)) and contains(area, (262.0, 300.0))
    assert field.route((300.0, 300.0)) is None  # the pointer on the friend: not in the area
    total, points, _ = field.route((300.0, 322.0))  # at the edge of the hole: the nearest place beside the friend
    assert np.linalg.norm(points[-1] - (300.0, 300.0)) == pytest.approx(16.0 + 20.0, abs=2.0)


def test_no_stop_on_an_open_hatchway_or_breach_point():
    # A breach gap of 35 mm: a 25 mm base fits in it, but it cannot stop there.
    shapes = [Shape("wall", np.array([[0, 298], [600, 298], [600, 302], [0, 302]], float)), Shape("breach", np.array([[282.5, 298], [317.5, 298], [317.5, 302], [282.5, 302]], float), open=True)]
    rects, gaps = obstacles(shapes), open_gaps(shapes)
    field = move_field((300.0, 260.0), 25.0, 6, rects, gaps)
    assert field.cost((300.0, 300.0)) is None and field.cost((300.0, 288.0)) is None  # on the gap, or the rim on it
    assert field.cost((300.0, 284.0)) == 1 and field.cost((300.0, 330.0)) == 4  # before it, and through it (70 mm = 3 in, and 1)
    area = field.outline()
    for x in (285.0, 300.0, 315.0):
        assert not contains(area, (x, 300.0))  # the gap is not in the area
    assert contains(area, (300.0, 294.0)) and contains(area, (300.0, 306.0))


def test_avoid_keeps_the_rim_one_inch_from_the_enemy_rim():
    enemy = ((300.0, 300.0), 40.0)
    start, goal = (200.0, 300.0), (400.0, 300.0)
    keep = 16.0 + 20.0 + MM_PER_INCH  # centre to centre
    plain = move_field(start, D, 9, enemies=[enemy])
    assert plain.cost((300.0 - keep + 6.0, 300.0)) is not None  # in the 1 in zone: a melee
    field = move_field(start, D, 9, enemies=[enemy], avoid_mm=MM_PER_INCH)
    assert field.cost((300.0 - keep + 3.0, 300.0)) is None and field.cost((300.0 - keep - 2.0, 300.0)) is not None
    total, points, costs = field.route(goal)
    assert total == sum(costs) and len(points) >= 3 and seg_dist(points, enemy[0]) > keep - 2.0
    area = field.outline()
    assert not contains(area, (300.0, 300.0 - 20.0 - MM_PER_INCH + 3.0)) and contains(area, (300.0, 300.0 - 20.0 - MM_PER_INCH - 3.0))


def test_avoid_does_not_lock_a_base_that_starts_in_melee():
    enemy = ((300.0, 300.0), 40.0)
    field = move_field((250.0, 300.0), D, 3, enemies=[enemy], avoid_mm=MM_PER_INCH)  # rims 14 mm apart
    assert field.cost((200.0, 300.0)) == 2 and field.cost((262.0, 300.0)) == 1
    assert field.cost((270.0, 300.0)) is None  # the bases would overlap
