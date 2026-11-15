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


def test_wall_near_the_shooter_base_does_not_obscure():
    wall = wall_x(ORIGIN[0] + 20.0)
    assert shoot.aim(ORIGIN, TARGET, D, wall).obscured  # no shooter base: only the pointer
    shot = shoot.aim(ORIGIN, TARGET, D, wall, shooter=(ORIGIN, 32.0))
    assert not shot.obscured and not shot.in_cover and len(shot.pieces) == 1


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
