"""Saved game state: a folder with the whole state of a `LiveSession`.

```
<folder>/state.json       the game state (schema `SCHEMA`)
<folder>/frame.png        camera frame of the snapshot (restore detects the bases on it again)
<folder>/empty_board.png  the empty-board reference (the state names this file)
```

No Qt. `capture` and `write` read a session, `read` and `apply` bring a saved
state back. The session writes the auto-save through `LiveSession.autosave`.

Every attribute of a `LiveSession` must be in `SAVED` or in `NOT_SAVED`.
`tests/test_state.py` checks this, so a new state attribute cannot be
forgotten: add it to the schema (and to `capture` and `apply`), or give the
reason why it is not state.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import score as sc
from .circles import Circle
from .ident import Ident

SCHEMA = 1
STATE_FILE = "state.json"
FRAME_FILE = "frame.png"
REFERENCE_FILE = "empty_board.png"

# Plain true/false view switches of the session. Saved and restored by name.
VIEW_FLAGS = ("top_down", "grid", "terrain", "zones", "keys", "light_shown", "marks", "around_walls", "movement", "avoid_melee")

# Attribute of `LiveSession` -> key of `state.json` that holds it.
SAVED: dict[str, str] = {
    **{name: f"view.{name}" for name in VIEW_FLAGS},
    "minimap": "view.minimap",
    "frozen": "view.frozen",
    "win": "view.win_top_down",  # BoardWindow.top_down
    "team_slots": "team_slots",
    "in_play": "in_play",
    "positions": "positions",
    "models": "models",
    "model_locked": "model_locked",
    "terrain_type": "terrain_type",
    "map_layout": "map_layout",
    "opened": "opened",
    "light_poses": "light_poses",
    "loose_light": "loose_light",
    "wounds": "wounds",
    "scores": "scores",
    "turn": "turn",
    "crit_op": "crit_op",
    "ui_layout": "ui_layout",
    "snap": "files.frame",
    "result": "snapshot.bases",
    "session": "snapshot.idents",
    "reference_frame": "files.reference",
    "selected": "selection.selected",
    "selected_operative": "selection.selected_operative",
    "target": "selection.target",
    "ring_in": "selection.ring_in",
    "selected_opening": "selection.selected_opening",
    "selected_light": "selection.selected_light",
    "new_base": "selection.new_base",
    "shooting": "selection.shooting",
}

# Attribute -> why it is not saved. Derived values are built again from the
# saved ones; the rest belongs to this run only (devices, caches, a pending click).
NOT_SAVED: dict[str, str] = {
    "cfg": "config file",
    "board": "camera tracking of this run",
    "lens": "camera tracking of this run",
    "pipe": "built from the config; its reference comes from `reference_frame`",
    "reference_path": "path option of this run",
    "snapshot_dir": "path option of this run",
    "saves_dir": "path option of this run",
    "autosave_path": "the auto-save folder of this run",
    "autosave_enabled": "option of this run",
    "_save_cache": "what the auto-save wrote already",
    "last_positions": "old places for the move arrows of the next identification of this run",
    "moved_from": "move arrows of the last identification, shown until the names are confirmed",
    "_paused_measure": "ring, second base and movement that \"Shoot at\" paused; not saved with the game",
    "_save_problem": "last auto-save error text",
    "detector": "camera tracking of this run",
    "monitor": "camera tracking of this run",
    "clock": "run",
    "notice": "message of the moment",
    "teams": "derived from `team_slots` (or the option of this run)",
    "gallery_root": "path option of this run",
    "record_photos": "option of this run (--image-source: no gallery photos)",
    "team_choices": "folder listing",
    "_team_ops": "cache of team files",
    "_team_infos": "cache of team files",
    "gallery": "loaded again from the team slots",
    "gallery_tried": "loaded again from the team slots",
    "_shot": "cache",
    "hover_mm": "pointer position",
    "move_locked": "needs the pointer position, which is not saved; a loaded game starts unlocked",
    "shot_locked": "needs the pointer position, which is not saved; a loaded game starts unlocked",
    "_move_field": "cache",
    "_move_route": "cache",
    "_navigator": "cache",
    "_wall_path": "cache",
    "_wall_range": "cache",
    "_ring_bases": "cache",
    "maps_root": "path option of this run",
    "pieces_root": "path option of this run",
    "terrain_map": "read from the map file of `map_layout`",
    "pieces": "read from the piece files of `terrain_type`",
    "locating_light": "pending click",
    "_light_version": "cache key",
    "_locate_image": "pending click",
    "_light_crop_cache": "cache",
    "maps": "folder listing",
    "terrain_shapes": "built from the layout, `light_poses` and `loose_light`",
    "openings": "built from the layout",
    "busy": "run",
    "camera_up": "camera of this run",
    "on_busy": "callback of the front end",
    "_preload": "model load of this run",
    "_task_progress": "run",
    "on_progress": "callback of the front end",
    "_preload_thread": "model load of this run",
    "frame": "newest camera frame; the snapshot frame is saved",
    "obs": "camera tracking of this run",
    "report": "camera tracking of this run",
    "view": "camera tracking of this run",
    "last_status": "camera tracking of this run",
    "image": "picture of the moment, drawn again (the Save screenshot action keeps pictures)",
    "detect_ms": "measurement of this run",
    "meter": "measurement of this run",
    "adding": "pending click",
    "right_click": "what the last right click did, for the front end",
    "shown": "rendered picture",
    "image_region": "where the zoomed picture sits, drawn again",
    "_xf": "transform of the render in progress",
    "dirty": "render flag",
    "_handlers": "action table",
}


class StateError(Exception):
    """A saved state that cannot be read or does not fit this app."""


def _num(value: float) -> float | None:
    """JSON number: None for -inf (a missing score)."""
    return None if value is None or not math.isfinite(value) else float(value)


def _unnum(value: float | None) -> float:
    return float("-inf") if value is None else float(value)


def _point(p) -> list[float]:
    return [float(p[0]), float(p[1])]


def _circle_to_json(c: Circle) -> dict[str, Any]:
    return {"center_mm": _point(c.center_mm), "diameter_mm": float(c.diameter_mm), "fit_diameter_mm": float(c.fit_diameter_mm), "coverage": float(c.coverage), "rms_mm": float(c.rms_mm), "contrast": float(c.contrast)}


def _circle_from_json(d: dict[str, Any]) -> Circle:
    return Circle(np.array(d["center_mm"], float), float(d["diameter_mm"]), float(d["fit_diameter_mm"]), float(d["coverage"]), float(d["rms_mm"]), float(d["contrast"]))


def _ident_to_json(x: Ident) -> dict[str, Any]:
    return {"label": x.label, "score": _num(x.score), "margin": _num(x.margin), "candidates": [[lab, _num(v)] for lab, v in x.candidates], "sure": x.sure, "confirmed": x.confirmed, "masked": x.masked}


def _ident_from_json(d: dict[str, Any]) -> Ident:
    return Ident(d["label"], _unnum(d["score"]), _unnum(d["margin"]), [(lab, _unnum(v)) for lab, v in d["candidates"]], sure=bool(d["sure"]), confirmed=bool(d["confirmed"]), masked=bool(d["masked"]))


def capture(s) -> dict[str, Any]:
    """The state of session `s` as a JSON-safe dict."""
    s._update_kill_scores()
    snapshot = None
    if s.snap is not None and s.result is not None:
        snapshot = {"bases": [_circle_to_json(c) for c in s._found()], "idents": None if s.session is None else [_ident_to_json(x) for x in s.session.idents]}
    return {
        "schema": SCHEMA,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ui_layout": s.ui_layout,
        "files": {"frame": FRAME_FILE if snapshot is not None else None, "reference": REFERENCE_FILE if s.reference_frame is not None else None},
        "team_slots": list(s.team_slots),
        "in_play": [None if x is None else list(x) for x in s.in_play],
        "positions": {k: _point(v) for k, v in sorted(s.positions.items())},
        "models": dict(sorted(s.models.items())),
        "model_locked": sorted(s.model_locked),
        "terrain_type": s.terrain_type,
        "map_layout": s.map_layout,
        "opened": sorted(s.opened),
        "light_poses": [[k, _point(c), float(a)] for k, (c, a) in sorted(s.light_poses.items())],
        "loose_light": [[key, _point(c), float(a)] for key, c, a in s.loose_light],
        "wounds": dict(sorted(s.wounds.items())),
        "scores": [x.to_state() for x in s.scores],
        "turn": s.turn,
        "crit_op": s.crit_op,
        "view": {**{name: bool(getattr(s, name)) for name in VIEW_FLAGS}, "minimap": bool(s.minimap), "frozen": bool(s.frozen), "win_top_down": bool(s.win.top_down)},
        "snapshot": snapshot,
        "selection": {name: getattr(s, name) for name in ("selected", "selected_operative", "target", "ring_in", "selected_opening", "selected_light", "new_base")} | {"shooting": bool(s.shooting)},
    }


# --- files ----------------------------------------------------------------------


def _write_bytes(path: Path, data: bytes) -> None:
    """Write through a temporary file: a crash never leaves a half file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _write_png(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise OSError(f"cannot encode {path.name}")
    _write_bytes(path, buf.tobytes())


def write(s, folder: Path | str, cache: dict[str, Any] | None = None) -> Path:
    """Save session `s` in `folder` (made when missing; an old save there is
    replaced). `cache` (the auto-save keeps one per folder) skips the frame
    and the reference when they did not change since the last write."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cache = {} if cache is None else cache
    data = capture(s)
    for key, name, image in (("snap", FRAME_FILE, s.snap if data["files"]["frame"] else None), ("reference_frame", REFERENCE_FILE, s.reference_frame)):
        path = folder / name
        if image is None:
            path.unlink(missing_ok=True)
            cache.pop(key, None)
        elif cache.get(key) is not image or not path.is_file():
            _write_png(path, image)
            cache[key] = image
    _write_bytes(folder / STATE_FILE, (json.dumps(data, indent=2, allow_nan=False) + "\n").encode())  # last: a state file means a complete save
    return folder


def read(path: Path | str) -> tuple[dict[str, Any], Path]:
    """The state dict of a save folder (or of its `state.json`), and the folder.
    A missing or bad file, or another schema, is a StateError."""
    path = Path(path)
    folder = path.parent if path.is_file() else path
    file = folder / STATE_FILE
    try:
        data = json.loads(file.read_text())
    except FileNotFoundError:
        raise StateError(f"no {STATE_FILE} in {folder}") from None
    except (OSError, ValueError) as exc:
        raise StateError(f"{file}: {exc}") from None
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise StateError(f"{file}: schema {data.get('schema') if isinstance(data, dict) else None!r}, this app reads schema {SCHEMA}")
    return data, folder


def _image(folder: Path, name: str | None) -> np.ndarray | None:
    if not name:
        return None
    img = cv2.imread(str(folder / name), cv2.IMREAD_COLOR)
    if img is None:
        raise StateError(f"cannot read {folder / name}")
    return img


def apply(s, path: Path | str) -> None:
    """Bring the saved state at `path` back into session `s`. A problem that
    stops the restore is a StateError (the session may be half restored: start
    it again). A part that does not fit (a team or a layout that is gone) gives
    a notice and the restore goes on."""
    data, folder = read(path)
    try:
        _apply(s, data, folder)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise StateError(f"{folder}: bad state: {exc!r}") from None


def _apply(s, data: dict[str, Any], folder: Path) -> None:
    files = data["files"]
    ref = _image(folder, files.get("reference"))
    s.reference_frame = None
    s.pipe.clear_reference()
    if ref is not None:
        try:
            s.pipe.set_reference(ref)
            s.reference_frame = ref
        except ValueError as exc:
            s.notice.error(f"saved reference not used: {exc}")
    # Teams, then the roster: `_set_team` resets the roster of a side.
    for slot, name in enumerate(data["team_slots"]):
        s._set_team(slot, name)
    s.in_play = [None if x is None else list(x) for x in data["in_play"]]
    s._update_roster()
    # The layout: loading it resets the openings and the light terrain poses.
    if data["terrain_type"] is not None:
        s._set_terrain_type(data["terrain_type"])
    s._set_map_layout(data["map_layout"])
    s.opened = {i for i in data["opened"] if 0 <= i < len(s.openings)}
    s.light_poses = {int(k): ((c[0], c[1]), float(a)) for k, c, a in data["light_poses"] if s.terrain_map is not None and 0 <= k < len(s.terrain_map.light)}
    s.loose_light = [(key, (c[0], c[1]), float(a)) for key, c, a in data["loose_light"] if key in s.pieces]
    if s.terrain_map is not None or s.loose_light:
        s._build_shapes()
    # Game values.
    s.positions = {k: np.array(v, float) for k, v in data["positions"].items()}
    s.models = {str(k): str(v) for k, v in data.get("models", {}).items()}  # older saves have no models
    s.model_locked = {str(k) for k in data.get("model_locked", [])}
    s.wounds = {k: int(v) for k, v in data["wounds"].items()}
    if len(data["scores"]) != len(s.scores):
        raise ValueError(f"{len(data['scores'])} scores, {len(s.scores)} sides")
    s.scores = [sc.Score.from_state(x) for x in data["scores"]]
    s.turn = int(data["turn"])
    s.crit_op = data["crit_op"]
    s.ui_layout = data["ui_layout"]
    # The snapshot: the bases on the saved frame, the names that were saved.
    snap = data["snapshot"]
    frame = _image(folder, files.get("frame"))
    s._close_snapshot()
    if snap is not None and frame is not None:
        s.restore_snapshot(frame, [_circle_from_json(c) for c in snap["bases"]], None if snap["idents"] is None else [_ident_from_json(x) for x in snap["idents"]])
    # Selection and view last: the snapshot changes both.
    sel = data["selection"]
    found = len(s._found())
    s._set_selected(sel["selected"] if sel["selected"] is not None and sel["selected"] < found else None)
    s.selected_operative = sel["selected_operative"]
    s.target = sel["target"] if sel["target"] is not None and sel["target"] < found else None
    s.ring_in = sel["ring_in"]
    s.selected_opening = sel["selected_opening"] if sel["selected_opening"] is not None and sel["selected_opening"] < len(s.openings) else None
    s.selected_light = sel["selected_light"] if sel["selected_light"] is not None and sel["selected_light"] < len(s._light_state()) else None
    s.new_base = sel["new_base"] if sel["new_base"] is not None and sel["new_base"] < found else None
    view = data["view"]
    for name in VIEW_FLAGS:
        setattr(s, name, bool(view[name]))
    s.minimap = bool(view["minimap"]) and s.map_layout is not None
    s.frozen = bool(view["frozen"]) and s.result is not None and not s.minimap
    s.win.top_down = bool(view["win_top_down"])
    s.shooting = bool(sel["shooting"]) and s.selected is not None
    s.dirty = True

