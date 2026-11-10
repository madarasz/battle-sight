"""Rings and rim-to-rim distances."""

from __future__ import annotations

import numpy as np
import pytest

from battlesight.measure import MM_PER_INCH, base_at, rim_gap, ring_radius_mm


def test_ring_radius_is_base_radius_plus_inches():
    assert ring_radius_mm(32.0, 6) == pytest.approx(16.0 + 6 * 25.4)
    assert ring_radius_mm(50.0, 1) == pytest.approx(25.0 + MM_PER_INCH)


def test_rim_gap_and_end_points():
    gap, start, end = rim_gap((0.0, 0.0), 32.0, (100.0, 0.0), 40.0)
    assert gap == pytest.approx(64.0)
    assert start == pytest.approx((16.0, 0.0)) and end == pytest.approx((80.0, 0.0))


def test_rim_gap_on_a_diagonal_is_symmetric():
    forward = rim_gap((10.0, 20.0), 25.0, (110.0, 95.0), 28.0)
    back = rim_gap((110.0, 95.0), 28.0, (10.0, 20.0), 25.0)
    assert forward[0] == pytest.approx(125.0 - 12.5 - 14.0) == back[0]
    assert np.linalg.norm(forward[2] - forward[1]) == pytest.approx(forward[0])
    assert forward[1] == pytest.approx(back[2]) and forward[2] == pytest.approx(back[1])


@pytest.mark.parametrize("between", [32.0, 10.0, 0.0])
def test_touching_or_overlapping_bases_have_no_gap(between):
    gap, start, end = rim_gap((0.0, 0.0), 32.0, (between, 0.0), 32.0)
    assert gap == 0.0 and start == pytest.approx(end)


def test_base_at_takes_the_nearest_base_within_reach():
    centers = np.array([[100.0, 100.0], [140.0, 100.0]])
    assert base_at(centers, [32.0, 32.0], (112.0, 100.0)) == 0
    assert base_at(centers, [32.0, 32.0], (128.0, 100.0)) == 1
    assert base_at(centers, [32.0, 32.0], (100.0, 125.0)) is None
    assert base_at(np.zeros((0, 2)), [], (0.0, 0.0)) is None


def test_health_bar_length_follows_the_maximum_wounds():
    from battlesight.overlay import BLACK, HEALTH_MM_PER_WOUND, draw_health_bars

    def bar_width(current: int, maximum: int) -> tuple[int, int]:
        img = np.zeros((200, 400, 3), np.uint8)
        draw_health_bars(img, [((200.0, 150.0), 40.0, current, maximum)], lambda p: np.asarray(p, float).reshape(-1, 2))
        cols = np.flatnonzero((img != BLACK).any(axis=2).any(axis=0))
        green = np.flatnonzero(((img[..., 1] > 150) | (img[..., 2] > 200)).any(axis=0))  # unharmed or wounded fill
        return cols[-1] - cols[0], 0 if green.size == 0 else green[-1] - green[0]

    w15, _ = bar_width(15, 15)
    w19, _ = bar_width(19, 19)
    assert abs(w15 - 15 * HEALTH_MM_PER_WOUND) <= 2 and abs(w19 - 19 * HEALTH_MM_PER_WOUND) <= 2
    full, fill_full = bar_width(18, 18)
    half, fill_half = bar_width(14, 18)
    assert full == half and abs(fill_half - fill_full * 14 / 18) <= 2


@pytest.mark.parametrize("current, color", [(18, "HEALTH_UNHARMED"), (17, "HEALTH_WOUNDED"), (9, "HEALTH_WOUNDED"), (8, "HEALTH_INJURED"), (0, "HEALTH_INJURED")])
def test_health_bar_color_follows_the_wounds(current, color):
    from battlesight import overlay

    assert overlay.health_color(current, 18) == getattr(overlay, color)
    assert overlay.health_color(7, 15) == overlay.HEALTH_INJURED and overlay.health_color(8, 15) == overlay.HEALTH_WOUNDED  # 15 / 2 = 7.5
