"""Game score of one side: victory points (VP), Tac Op, Primary and command
points (CP). Pure state, no window.

Crit Op and Tac Op score in turning points 2 to 4, `SKULLS_PER_TP` skulls
each. Kill Op is one cell of `CATEGORY_MAX` skulls (later it comes from the
game state). Each category has `CATEGORY_MAX` VP or less. The Primary adds
half of the points of its category, rounded up; "hidden" adds nothing.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

APPROVED_OPS_FILE = Path(__file__).resolve().parents[2] / "data" / "approved-ops-2025.toml"
CATEGORIES = ("crit", "tac", "kill")
CATEGORY_NAMES = {"crit": "Crit Op", "tac": "Tac Op", "kill": "Kill Op"}
PRIMARIES = ("hidden", *CATEGORIES)
TURNING_POINTS = (2, 3, 4)  # turning points with score cells of Crit Op and Tac Op
CATEGORY_MAX = 6
SKULLS_PER_TP = CATEGORY_MAX // len(TURNING_POINTS)
CP_START = 3


def load_tac_ops(path: Path = APPROVED_OPS_FILE) -> dict[str, tuple[str, ...]]:
    """{archetype: Tac Op names} from `[tac_ops]` of the approved ops file."""
    with path.open("rb") as fh:
        return {k: tuple(v) for k, v in tomllib.load(fh)["tac_ops"].items()}


TAC_OPS = load_tac_ops()


def tac_ops_for(archetypes: tuple[str, ...] | list[str]) -> list[str]:
    """Tac Ops of a team: the ops of each of its archetypes, in order."""
    return [op for a in archetypes for op in TAC_OPS.get(a, ())]


def skull_value(current: int, skull: int) -> int:
    """New cell value after a click on skull `skull` (1 = first): the cell
    fills up to that skull; a click on the last lit skull turns it off."""
    return skull - 1 if skull == current else skull


@dataclass
class Score:
    """Score of one side of the board."""

    cells: dict[str, list[int]] = field(default_factory=lambda: {c: [0] * len(TURNING_POINTS) for c in ("crit", "tac")} | {"kill": [0]})
    tac_op: str | None = None
    primary: str = "hidden"
    cp: int = CP_START

    def cell_max(self, category: str) -> int:
        return CATEGORY_MAX if category == "kill" else SKULLS_PER_TP

    def cell_index(self, category: str, tp: int | None) -> int:
        """Index of a cell. Crit and Tac need a turning point of
        TURNING_POINTS, Kill none. Else a ValueError."""
        if category not in CATEGORIES:
            raise ValueError(f"category must be one of {list(CATEGORIES)}, not {category!r}")
        if category == "kill":
            if tp is not None:
                raise ValueError("Kill Op has no turning point cells")
            return 0
        if tp not in TURNING_POINTS:
            raise ValueError(f"turning point must be one of {list(TURNING_POINTS)}, not {tp!r}")
        return TURNING_POINTS.index(tp)

    def click_skull(self, category: str, tp: int | None, skull: int) -> None:
        """A click on one skull of a cell. A bad value is a ValueError."""
        i = self.cell_index(category, tp)
        top = self.cell_max(category)
        if isinstance(skull, bool) or not isinstance(skull, int) or not 1 <= skull <= top:
            raise ValueError(f"skull must be a whole number from 1 to {top}, not {skull!r}")
        cells = self.cells[category]
        cells[i] = skull_value(cells[i], skull)

    def points(self, category: str) -> int:
        return min(sum(self.cells[category]), CATEGORY_MAX)

    def set_primary(self, value: str) -> None:
        if value not in PRIMARIES:
            raise ValueError(f"primary must be one of {list(PRIMARIES)}, not {value!r}")
        self.primary = value

    def primary_bonus(self) -> int:
        return 0 if self.primary == "hidden" else math.ceil(self.points(self.primary) / 2)

    def total(self) -> int:
        return sum(self.points(c) for c in CATEGORIES) + self.primary_bonus()

    def add_cp(self, delta: int) -> None:
        self.cp = max(0, self.cp + delta)

    def to_dict(self, tac_ops: list[str]) -> dict[str, Any]:
        """JSON-safe state. `tac_ops` are the choices for this side's team."""
        return {
            "cells": {c: list(v) for c, v in self.cells.items()},
            "points": {c: self.points(c) for c in CATEGORIES},
            "tac_op": self.tac_op,
            "tac_ops": list(tac_ops),
            "primary": self.primary,
            "primary_bonus": self.primary_bonus(),
            "cp": self.cp,
            "total": self.total(),
        }
