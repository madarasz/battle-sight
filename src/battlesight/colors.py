"""UI colours from `data/color-settings.toml`, read once at import.

Each constant names a function, not a colour. The constant of the key
`team_1` is `TEAM_1_COLOR`. Colours for OpenCV drawing are BGR tuples.
Colours of the `[window]` table are Qt colour strings: `#rrggbb`, or
`#aarrggbb` for a file value `#rrggbbaa`. No Qt import here.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

COLOR_FILE = Path(__file__).resolve().parents[2] / "data" / "color-settings.toml"
QT_TABLE = "window"  # table of Qt colours; all other tables are OpenCV BGR colours

_HEX = re.compile(r"#([0-9a-fA-F]{6})([0-9a-fA-F]{2})?")


class ColorError(ValueError):
    """A bad or incomplete colour file."""


def parse_hex(text: str, alpha: bool = False) -> tuple[int, ...]:
    """`#rrggbb` -> (r, g, b). With `alpha`, also `#rrggbbaa` -> (r, g, b, a)."""
    m = _HEX.fullmatch(text.strip()) if isinstance(text, str) else None
    if m is None or (m.group(2) and not alpha):
        raise ColorError(f"bad colour {text!r}: use \"#rrggbb\"" + (" or \"#rrggbbaa\"" if alpha else ""))
    rgb = tuple(int(m.group(1)[k : k + 2], 16) for k in (0, 2, 4))
    return rgb + (int(m.group(2), 16),) if m.group(2) else rgb


def load(path: Path = COLOR_FILE) -> dict[str, tuple[int, int, int] | str]:
    """Key -> colour: BGR tuple, or a Qt colour string for the `[window]` table."""
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ColorError(f"{path}: {e}") from e
    out: dict[str, tuple[int, int, int] | str] = {}
    for table, values in data.items():
        if not isinstance(values, dict):
            raise ColorError(f"{path}: {table!r} is not a table")
        for key, text in values.items():
            if key in out:
                raise ColorError(f"{path}: key {key!r} is in two tables")
            try:
                if table == QT_TABLE:
                    rgba = parse_hex(text, alpha=True)
                    r, g, b = rgba[:3]
                    out[key] = f"#{rgba[3]:02x}{r:02x}{g:02x}{b:02x}" if len(rgba) == 4 else f"#{r:02x}{g:02x}{b:02x}"
                else:
                    r, g, b = parse_hex(text)
                    out[key] = (b, g, r)
            except ColorError as e:
                raise ColorError(f"{path}: [{table}] {key}: {e}") from None
    return out


_c = load()
_used: set[str] = set()


def _get(key: str):
    if key not in _c:
        raise ColorError(f"{COLOR_FILE}: key {key!r} is missing")
    _used.add(key)
    return _c[key]


# --- status strips and messages on the video
STATUS_OK_COLOR = _get("status_ok")
STATUS_HINT_COLOR = _get("status_hint")
STATUS_ERROR_COLOR = _get("status_error")
BANNER_BACK_COLOR = _get("banner_back")

# --- shared drawing
OUTLINE_COLOR = _get("outline")
LABEL_TEXT_COLOR = _get("label_text")
POINTER_COLOR = _get("pointer")
POINTER_CROSS_COLOR = _get("pointer_cross")
POINTER_CROSS_EDGE_COLOR = _get("pointer_cross_edge")

# --- board corners and grid
BOARD_FOUND_COLOR = _get("board_found")
TAG_MOVED_COLOR = _get("tag_moved")
TAG_ORIGIN_COLOR = _get("tag_origin")
BOARD_CORNER_COLOR = _get("board_corner")
CORNER_LOST_COLOR = _get("corner_lost")
GRID_LINE_COLOR = _get("grid_line")
GRID_BOARD_EDGE_COLOR = _get("grid_board_edge")

# --- bases, names and teams
BASE_COLOR = _get("base")
NAME_SURE_COLOR = _get("name_sure")
NAME_UNSURE_COLOR = _get("name_unsure")
NAME_UNKNOWN_COLOR = _get("name_unknown")
TEAM_1_COLOR = _get("team_1")
TEAM_2_COLOR = _get("team_2")
NO_TEAM_COLOR = _get("no_team")
SELECTED_COLOR = _get("selected")
HEALTH_EMPTY_COLOR = _get("health_empty")
HEALTH_UNHARMED_COLOR = _get("health_unharmed")
HEALTH_WOUNDED_COLOR = _get("health_wounded")
HEALTH_INJURED_COLOR = _get("health_injured")

# --- measurement
MEASURE_COLOR = _get("measure")
RANGE_ZONE_COLOR = _get("range_zone")
MOVE_COLOR = _get("move")
MOVE_MELEE_COLOR = _get("move_melee")
TARGETING_COLOR = _get("targeting")
INTERVENING_COLOR = _get("intervening")
MARK_COLOR = _get("mark")
MARK_BACK_COLOR = _get("mark_back")

# --- terrain
WALL_COLOR = _get("wall")
PILLAR_COLOR = _get("pillar")
HATCHWAY_COLOR = _get("hatchway")
BREACH_COLOR = _get("breach")
LIGHT_TERRAIN_COLOR = _get("light_terrain")
DROP_ZONE_ORANGE_COLOR = _get("drop_zone_orange")
DROP_ZONE_GREY_COLOR = _get("drop_zone_grey")
OBJECTIVE_COLOR = _get("objective")
TELEPORTER_COLOR = _get("teleporter")
MINIMAP_BACK_COLOR = _get("minimap_back")
MINIMAP_BOARD_COLOR = _get("minimap_board")

# --- `map` command image
MAP_PAPER_COLOR = _get("map_paper")
MAP_BOARD_COLOR = _get("map_board")
MAP_GRID_COLOR = _get("map_grid")
MAP_TEXT_COLOR = _get("map_text")

# --- `enroll` and `calibrate` windows
ENROLL_CANDIDATE_COLOR = _get("enroll_candidate")
CHARUCO_CORNER_COLOR = _get("charuco_corner")

# --- developer tools
BENCH_MATCH_COLOR = _get("bench_match")
BENCH_WRONG_SIZE_COLOR = _get("bench_wrong_size")
BENCH_FALSE_COLOR = _get("bench_false")
BENCH_MISSED_COLOR = _get("bench_missed")
NOT_BASE_COLOR = _get("not_base")
ANNOTATE_CLEAR_RIM_COLOR = _get("annotate_clear_rim")
ANNOTATE_HIDDEN_RIM_COLOR = _get("annotate_hidden_rim")
ANNOTATE_PROPOSAL_COLOR = _get("annotate_proposal")
ANNOTATE_POINT_COLOR = _get("annotate_point")
ANNOTATE_FIT_COLOR = _get("annotate_fit")

# --- Qt window (strings for QColor and style sheets)
MESSAGE_NONE_COLOR = _get("message_none")
MESSAGE_OK_COLOR = _get("message_ok")
MESSAGE_WARN_COLOR = _get("message_warn")
MESSAGE_ERROR_COLOR = _get("message_error")
MESSAGE_INFO_COLOR = _get("message_info")
MODE_LIVE_COLOR = _get("mode_live")
MODE_SNAPSHOT_COLOR = _get("mode_snapshot")
MODE_MINIMAP_COLOR = _get("mode_minimap")
SHOT_OBSCURED_COLOR = _get("shot_obscured")
SHOT_IN_COVER_COLOR = _get("shot_in_cover")
VIDEO_BACK_COLOR = _get("video_back")
BADGE_BACK_COLOR = _get("badge_back")
BADGE_TEXT_COLOR = _get("badge_text")
SELECTED_ROW_COLOR = _get("selected_row")
SKULL_ON_COLOR = _get("skull_on")
SKULL_OFF_COLOR = _get("skull_off")
SCORE_GRID_COLOR = _get("score_grid")

if _c.keys() - _used:
    raise ColorError(f"{COLOR_FILE}: unknown keys {sorted(_c.keys() - _used)}")

TEAM_COLORS = (TEAM_1_COLOR, TEAM_2_COLOR)  # team slot 0 (left) and 1 (right)


def qt_hex(bgr: tuple[int, int, int]) -> str:
    """Qt colour string `#rrggbb` of an OpenCV BGR colour."""
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"
