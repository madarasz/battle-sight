"""Game score of one side: skulls, Primary bonus, command points."""

from __future__ import annotations

import pytest

from battlesight import score as sc


def test_tac_ops_come_from_the_approved_ops_file():
    assert set(sc.TAC_OPS) == {"Recon", "Security", "Seek and Destroy", "Infiltration"}
    assert all(len(ops) == 3 for ops in sc.TAC_OPS.values())
    ops = sc.tac_ops_for(["Seek and Destroy", "Security"])
    assert ops == [*sc.TAC_OPS["Seek and Destroy"], *sc.TAC_OPS["Security"]]
    assert sc.tac_ops_for([]) == []


def test_skull_click_fills_up_to_it_and_the_last_one_turns_off():
    assert sc.skull_value(0, 2) == 2
    assert sc.skull_value(2, 2) == 1
    assert sc.skull_value(1, 1) == 0
    assert sc.skull_value(2, 1) == 1


def test_cells_points_and_total():
    s = sc.Score()
    assert s.cp == 3 and s.primary == "hidden" and s.total() == 0
    assert sc.SKULLS_PER_TP * len(sc.TURNING_POINTS) == sc.CATEGORY_MAX
    for tp in sc.TURNING_POINTS:
        s.click_skull("crit", tp, 2)
    s.click_skull("tac", 3, 1)
    s.click_skull("kill", None, 5)
    assert s.cells == {"crit": [2, 2, 2], "tac": [0, 1, 0], "kill": [5]}
    assert [s.points(c) for c in sc.CATEGORIES] == [6, 1, 5] and s.total() == 12
    s.click_skull("kill", None, 5)
    assert s.points("kill") == 4


@pytest.mark.parametrize("primary,bonus", [("hidden", 0), ("crit", 3), ("tac", 1), ("kill", 3)])
def test_primary_adds_half_rounded_up(primary, bonus):
    s = sc.Score()
    s.cells = {"crit": [2, 2, 2], "tac": [1, 0, 0], "kill": [5]}
    s.set_primary(primary)
    assert s.primary_bonus() == bonus and s.total() == 12 + bonus


def test_bad_values_are_value_errors():
    s = sc.Score()
    with pytest.raises(ValueError):
        s.click_skull("crit", 1, 1)  # no TP1 cell
    with pytest.raises(ValueError):
        s.click_skull("kill", 2, 1)  # Kill has one cell
    with pytest.raises(ValueError):
        s.click_skull("crit", 2, 3)  # 2 skulls per TP
    with pytest.raises(ValueError):
        s.click_skull("vp", 2, 1)
    with pytest.raises(ValueError):
        s.set_primary("CritOp")
    assert s.total() == 0


def test_command_points_never_go_below_zero():
    s = sc.Score()
    for _ in range(5):
        s.add_cp(-1)
    assert s.cp == 0
    s.add_cp(2)
    assert s.to_dict([])["cp"] == 2
