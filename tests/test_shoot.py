import numpy as np
import pytest

from battlesight import shoot
from battlesight.measure import MM_PER_INCH

ORIGIN, TARGET, D = (100.0, 100.0), (300.0, 100.0), 32.0


def wall_x(x: float, y0: float = 50.0, y1: float = 150.0) -> np.ndarray:
    """A thin wall across the line of fire at x."""
    return np.array([[x - 2.0, y0, x + 2.0, y1]])


def test_tangent_points_touch_the_rim_at_a_right_angle():
    t = shoot.tangent_points(ORIGIN, TARGET, D)
    assert t.shape == (2, 2)
    for p in t:
        assert np.linalg.norm(p - TARGET) == pytest.approx(D / 2)
        assert np.dot(p - ORIGIN, p - TARGET) == pytest.approx(0.0, abs=1e-9)
    assert t[0][1] != t[1][1]  # one each side of the centre line
    outline = shoot.facing_outline(ORIGIN, TARGET, D)
    assert np.allclose(outline[0], ORIGIN) and np.allclose(outline[1], t[0]) and np.allclose(outline[-1], t[1])
    assert np.all(outline[1:, 0] <= TARGET[0] + 1e-9)  # the facing rim only
    assert shoot.tangent_points((305.0, 100.0), TARGET, D) is None and shoot.aim((305.0, 100.0), TARGET, D, wall_x(200.0)) is None


def test_wall_between_obscures():
    shot = shoot.aim(ORIGIN, TARGET, D, wall_x(200.0))
    assert shot.obscured and not shot.in_cover and len(shot.pieces) == 1


def test_wall_near_the_target_gives_cover():
    x = TARGET[0] - D / 2 - 15.0  # 15 mm from the target rim
    shot = shoot.aim(ORIGIN, TARGET, D, wall_x(x))
    assert shot.in_cover and not shot.obscured


def test_a_long_wall_can_give_cover_and_obscure():
    rect = np.array([[150.0, 98.0, 280.0, 102.0]])  # along the line of fire, into the target zone
    shot = shoot.aim(ORIGIN, TARGET, D, rect)
    assert shot.in_cover and shot.obscured


def test_wall_near_the_shooter_does_not_obscure():
    wall = wall_x(ORIGIN[0] + 20.0)  # faces 18 and 22 mm from the pointer: in 1 in of it
    near_pointer = shoot.aim(ORIGIN, TARGET, D, wall)
    assert not near_pointer.obscured and not near_pointer.in_cover and len(near_pointer.pieces) == 1
    assert not shoot.aim(ORIGIN, TARGET, D, wall, shooter=(ORIGIN, 32.0)).obscured
    farther = wall_x(ORIGIN[0] + 35.0)  # out of 1 in from the pointer, in 1 in of the base rim
    assert shoot.aim(ORIGIN, TARGET, D, farther).obscured
    assert not shoot.aim(ORIGIN, TARGET, D, farther, shooter=(ORIGIN, 32.0)).obscured


def test_terrain_off_the_lines_is_not_intervening():
    shot = shoot.aim(ORIGIN, TARGET, D, wall_x(200.0, 150.0, 250.0))
    assert len(shot.pieces) == 0 and not shot.obscured and not shot.in_cover
    assert len(shoot.aim(ORIGIN, TARGET, D, np.zeros((0, 4))).pieces) == 0


def test_the_lines_are_1_mm_wide():
    top = min(shoot.tangent_points(ORIGIN, TARGET, D), key=lambda p: p[1])  # end of the upper line
    y_line = ORIGIN[1] + (top[1] - ORIGIN[1]) * (200.0 - ORIGIN[0]) / (top[0] - ORIGIN[0])  # at x = 200

    def above(gap: float) -> np.ndarray:
        return np.array([[199.0, y_line - gap - 10.0, 201.0, y_line - gap]])

    assert shoot.aim(ORIGIN, TARGET, D, above(0.3)).obscured  # 0.3 mm off the line: touched
    assert not shoot.aim(ORIGIN, TARGET, D, above(1.3)).obscured  # 1.3 mm off the line


def test_zone_is_one_inch_from_the_rim():
    just_in = TARGET[0] - D / 2 - MM_PER_INCH + 3.0  # wall faces 1 mm and 5 mm from the zone edge
    just_out = TARGET[0] - D / 2 - MM_PER_INCH - 3.0
    assert shoot.aim(ORIGIN, TARGET, D, wall_x(just_in)).in_cover
    out = shoot.aim(ORIGIN, TARGET, D, wall_x(just_out))
    assert out.obscured and not out.in_cover


# --- light terrain and the 2 in rule ----------------------------------------------


def block(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


NEAR = TARGET[0] - D / 2 - 15.0  # 15 mm from the target rim: in its zone
NO_WALLS = np.zeros((0, 4))


def test_light_terrain_in_the_target_zone_gives_cover_and_never_obscures():
    shot = shoot.aim(ORIGIN, TARGET, D, NO_WALLS, light=[block(NEAR - 5, 90, NEAR + 5, 110)])
    assert shot.in_cover and not shot.obscured and shot.light == (0,) and shot.cover_by == "light" and len(shot.pieces) == 0
    mid = shoot.aim(ORIGIN, TARGET, D, NO_WALLS, light=[block(195, 50, 205, 150)])  # between, out of both zones
    assert not mid.in_cover and not mid.obscured and mid.light == () and mid.cover_by is None


def test_light_terrain_beside_the_lines_gives_no_cover():
    beside = block(NEAR - 5, 130, NEAR + 5, 150)  # in the zone, not in the lines
    behind = block(TARGET[0] + D / 2 + 5, 90, TARGET[0] + D / 2 + 15, 110)  # in the zone, behind the target
    shot = shoot.aim(ORIGIN, TARGET, D, NO_WALLS, light=[beside, behind])
    assert not shot.in_cover and shot.light == ()


def test_light_terrain_outline_counts_not_its_bounding_box():
    # An L around the lines: its box is in the lines, its area is not.
    y = TARGET[1]
    hook = np.array([[NEAR - 5, y - 40], [NEAR + 5, y - 40], [NEAR + 5, y + 40], [NEAR - 60, y + 40], [NEAR - 60, y + 30], [NEAR - 5, y + 30]], float)
    assert shoot.aim(ORIGIN, TARGET, D, NO_WALLS, light=[hook]).in_cover  # the upright crosses the lines
    arms = np.array([[NEAR - 60, y - 40], [NEAR + 5, y - 40], [NEAR + 5, y - 30], [NEAR - 50, y - 30], [NEAR - 50, y + 30], [NEAR + 5, y + 30], [NEAR + 5, y + 40], [NEAR - 60, y + 40]], float)
    c = shoot.aim(ORIGIN, TARGET, D, NO_WALLS, light=[arms])  # a C open to the target: the upright is out of the zone
    assert not c.in_cover and c.light == ()


def test_the_lines_touch_light_terrain_within_half_a_mm():
    top = min(shoot.tangent_points(ORIGIN, TARGET, D), key=lambda p: p[1])
    y_line = ORIGIN[1] + (top[1] - ORIGIN[1]) * (NEAR - ORIGIN[0]) / (top[0] - ORIGIN[0])
    assert shoot.aim(ORIGIN, TARGET, D, NO_WALLS, light=[block(NEAR - 1, y_line - 10.3, NEAR + 1, y_line - 0.3)]).in_cover
    assert not shoot.aim(ORIGIN, TARGET, D, NO_WALLS, light=[block(NEAR - 1, y_line - 11.5, NEAR + 1, y_line - 1.5)]).in_cover


def test_heavy_and_light_cover_together():
    shot = shoot.aim(ORIGIN, TARGET, D, wall_x(NEAR), light=[block(NEAR - 5, 90, NEAR + 5, 110), block(0, 0, 10, 10)])
    assert shot.in_cover and shot.cover_by == "both" and shot.light == (0,)
    assert shoot.aim(ORIGIN, TARGET, D, wall_x(NEAR)).cover_by == "heavy"


def test_no_cover_within_2_in_of_the_shooter():
    rim = TARGET[0] - D / 2
    cover = [block(rim - 12, 90, rim - 8, 110)]
    far = (rim - 2 * MM_PER_INCH - 2.0, TARGET[1])
    near = (rim - 2 * MM_PER_INCH + 2.0, TARGET[1])
    a = shoot.aim(far, TARGET, D, NO_WALLS, light=cover)
    assert a.in_cover and not a.too_near
    b = shoot.aim(near, TARGET, D, NO_WALLS, light=cover)
    assert not b.in_cover and b.too_near and b.light == () and b.cover_by is None
    # With a shooter base the distance is rim to rim: the pointer is far, the rim is near.
    base = ((far[0] + 10.0, TARGET[1]), 32.0)
    assert shoot.aim(far, TARGET, D, NO_WALLS, shooter=base, light=cover).too_near
    heavy = shoot.aim(near, TARGET, D, wall_x(rim - 10.0))
    assert not heavy.in_cover and heavy.too_near  # the rule is the same for heavy terrain
    assert not shoot.aim(near, TARGET, D, NO_WALLS).too_near  # nothing in the zone
