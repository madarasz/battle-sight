"""UI colours from data/color-settings.toml (`battlesight.colors`)."""

from __future__ import annotations

import re

import pytest

from battlesight import colors


def test_every_file_key_has_a_constant():
    names = {n for n in dir(colors) if n.endswith("_COLOR")}
    keys = set(colors.load())
    assert names == {f"{k.upper()}_COLOR" for k in keys}


def test_drawing_colors_are_bgr_and_window_colors_are_qt_strings():
    assert colors.TEAM_1_COLOR == (60, 60, 230)  # "#e63c3c" in the file
    assert colors.MESSAGE_OK_COLOR == "#2e9e4f"
    assert colors.BADGE_BACK_COLOR == "#aa000000"  # "#000000aa" -> Qt #aarrggbb
    assert colors.TEAM_COLORS == (colors.TEAM_1_COLOR, colors.TEAM_2_COLOR)
    for name in dir(colors):
        if name.endswith("_COLOR"):
            value = getattr(colors, name)
            assert (isinstance(value, str) and re.fullmatch(r"#([0-9a-f]{2})?[0-9a-f]{6}", value)) or (isinstance(value, tuple) and len(value) == 3), name


def test_parse_hex():
    assert colors.parse_hex("#ff8000") == (255, 128, 0)
    assert colors.parse_hex("#FF800040", alpha=True) == (255, 128, 0, 64)
    for bad in ("ff8000", "#ff80", "#ff800040", 12):
        with pytest.raises(colors.ColorError):
            colors.parse_hex(bad)


def test_bad_files_give_a_clear_error(tmp_path):
    path = tmp_path / "c.toml"
    for text, words in (
        ('[a]\nx = "red"\n', "[a] x"),
        ('[a]\nx = "#ffffff"\n[b]\nx = "#000000"\n', "two tables"),
        ("[a\n", str(path)),
        ('x = "#ffffff"\n', "not a table"),
    ):
        path.write_text(text)
        with pytest.raises(colors.ColorError, match=re.escape(words)):
            colors.load(path)
    with pytest.raises(colors.ColorError):
        colors.load(tmp_path / "missing.toml")
