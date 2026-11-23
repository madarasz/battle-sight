"""UI colours from data/color-settings.toml (`battlesight.colors`)."""

from __future__ import annotations

import re
import tomllib

import pytest

from battlesight import colors


def test_every_file_key_has_a_constant():
    names = {n for n in dir(colors) if n.endswith("_COLOR")}
    keys = set(colors.load())
    assert names == {f"{k.upper()}_COLOR" for k in keys}


def file_colors() -> dict[str, tuple[str, str]]:
    """Key -> (table, `#rrggbb[aa]` text) as written in data/color-settings.toml."""
    data = tomllib.loads(colors.COLOR_FILE.read_text())
    return {key: (table, text) for table, values in data.items() for key, text in values.items()}


def test_each_constant_is_the_file_value_as_bgr_or_qt_string():
    """The values come from the file: the test reads them, so that a colour
    change in the file never makes it fail."""
    for key, (table, text) in file_colors().items():
        value = getattr(colors, f"{key.upper()}_COLOR")
        r, g, b = (int(text[i : i + 2], 16) for i in (1, 3, 5))
        if table == colors.QT_TABLE:  # Qt: #rrggbb, or #aarrggbb for #rrggbbaa
            assert value == ("#" + text[7:9] + text[1:7] if len(text) == 9 else text).lower(), key
        else:  # OpenCV: BGR, and `qt_hex` gives the file text back
            assert value == (b, g, r), key
            assert colors.qt_hex(value) == text.lower(), key
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
