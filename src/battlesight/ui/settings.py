"""Reads and writes data/ui.toml: the layout choice, the layout options and
the last terrain map (`[map] terrain`, `[map] layout`; "" = no layout).

A bad file must not stop the app, so `load_ui_settings` never raises for its
contents. It gives the problem as text, and the window shows it as a notice.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_UI_FILE = Path(__file__).resolve().parents[3] / "data" / "ui.toml"



@dataclass(frozen=True)
class UiSettings:
    layout: str | None = None  # None: use the default layout
    options: dict[str, dict[str, Any]] = field(default_factory=dict)  # layout name -> its options
    problem: str | None = None  # text for a notice
    map_terrain: str | None = None  # last terrain type
    map_layout: str | None = None  # last map layout of that type

    def options_for(self, layout: str) -> dict[str, Any]:
        return self.options.get(layout, {})


def load_ui_settings(path: str | Path = DEFAULT_UI_FILE) -> UiSettings:
    """The saved settings. A missing file gives the defaults and no problem."""
    path = Path(path)
    try:
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
    except FileNotFoundError:
        return UiSettings()
    except (OSError, ValueError) as exc:  # tomllib.TOMLDecodeError is a ValueError
        return UiSettings(problem=f"{path.name} not used: {exc}")

    problems = []
    ui = raw.get("ui", {})
    layout = ui.get("layout") if isinstance(ui, dict) else None
    if not isinstance(ui, dict) or not isinstance(layout, str | None):
        problems.append("[ui] layout must be text")
        layout = None
    tables = raw.get("layouts", {})
    if not isinstance(tables, dict):
        problems.append("[layouts] must be a table")
        tables = {}
    options = {name: table for name, table in tables.items() if isinstance(table, dict)}
    problems += [f"[layouts.{name}] must be a table" for name in tables if name not in options]
    chosen = raw.get("map", {})
    terrain = chosen.get("terrain") if isinstance(chosen, dict) else None
    map_layout = chosen.get("layout") if isinstance(chosen, dict) else None
    if not isinstance(chosen, dict) or not isinstance(terrain, str | None) or not isinstance(map_layout, str | None):
        problems.append("[map] terrain and layout must be text")
        terrain = map_layout = None
    return UiSettings(layout, options, f"{path.name}: {', '.join(problems)}" if problems else None, terrain or None, map_layout or None)


def save_layout(path: str | Path, layout: str) -> None:
    """Set `[ui] layout` and keep all other text of the file, comments too.

    Raises ValueError and leaves the file as it is when the file is not valid
    TOML or the edit cannot set the value (for example a dotted `ui.layout`)."""
    save_values(path, "ui", {"layout": layout})


def save_map(path: str | Path, terrain: str | None, layout: str | None) -> None:
    """Set `[map] terrain` and `[map] layout` ("" for None), like `save_layout`."""
    save_values(path, "map", {"terrain": terrain or "", "layout": layout or ""})


def save_values(path: str | Path, table: str, values: dict[str, str]) -> None:
    """Set text values in one table and keep all other text of the file."""
    path = Path(path)
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    before = tomllib.loads(original)
    lines = original.splitlines()
    for key, value in values.items():
        lines = _with_value(lines, table, key, value)
    text = "\n".join(lines) + "\n"
    old = before.get(table, {})
    if not isinstance(old, dict) or tomllib.loads(text) != {**before, table: {**old, **values}}:
        raise ValueError(f"cannot set [{table}] {', '.join(values)} in {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _with_value(lines: list[str], table: str, key: str, value: str) -> list[str]:
    header_re = re.compile(rf"\s*\[{re.escape(table)}\]\s*(#.*)?$")
    key_re = re.compile(rf"\s*{re.escape(key)}\s*=")
    new_line = f"{key} = {json.dumps(value)}"
    header = next((i for i, line in enumerate(lines) if header_re.match(line)), None)
    if header is None:
        return [f"[{table}]", new_line, *([""] if lines else []), *lines]
    end = next((i for i in range(header + 1, len(lines)) if lines[i].lstrip().startswith("[")), len(lines))
    for i in range(header + 1, end):
        if key_re.match(lines[i]):
            return [*lines[:i], new_line, *lines[i + 1 :]]
    return [*lines[: header + 1], new_line, *lines[header + 1 :]]
