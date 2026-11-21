"""Saved game state: `battlesight.state`, the Save state action and the auto-save."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from battlesight import state as savestate
from battlesight.live import ACTIONS, NO_AUTOSAVE, LiveSession
from battlesight.score import DERIVED, Score

from .conftest import render_board
from .test_ident import ident_cfg
from .test_live import SMALL_MAP, base_of, make_session, named, named_session, named_template, set_wounds, settled, write_map  # noqa: F401


def games(tmp_path, named):
    """A named snapshot with a layout and game values that differ from the defaults."""
    write_map(tmp_path / "maps")
    set_wounds(tmp_path, {"Nob": 5, "Burna": 12})
    session, frame, state = named_session(tmp_path, named)
    session.dispatch("map.layout", name="Small")
    session.dispatch("base.select", index=base_of(state, "Nob"))
    session.dispatch("wounds.minus")
    session.dispatch("ring.3")
    session.dispatch("score.cp", slot=0, delta=-1)
    session.dispatch("score.skull", slot=0, category="crit", tp=2, skull=2)
    session.dispatch("score.primary", slot=1, value="tac")
    session.dispatch("game.turn", delta=1)
    session.dispatch("view.grid")
    session.dispatch("view.layout", name="inspector")
    return session, frame


def fresh(tmp_path, named, **over) -> LiveSession:
    cfg, frame = named
    return make_session(tmp_path / "again", cfg, frame, teams=("orks",), maps_root=tmp_path / "maps", gallery_root=tmp_path / "gallery", **over)


# --- the guard: no state attribute without a decision ---------------------------------


def test_every_session_attribute_is_saved_or_not_saved_with_a_reason(tmp_path, named):
    cfg, frame = named
    session = make_session(tmp_path, cfg, frame)
    names = set(vars(session))
    saved, skipped = set(savestate.SAVED), set(savestate.NOT_SAVED)
    assert not saved & skipped, f"in both lists of battlesight/state.py: {sorted(saved & skipped)}"
    new = names - saved - skipped
    assert not new, f"new LiveSession state {sorted(new)}: save it in battlesight/state.py (SAVED, capture, apply) or give a reason in NOT_SAVED"
    gone = (saved | skipped) - names
    assert not gone, f"battlesight/state.py names attributes that LiveSession does not have: {sorted(gone)}"
    assert all(savestate.NOT_SAVED.values())


def test_saved_keys_are_in_the_state_file(tmp_path, named):
    session, _ = games(tmp_path, named)
    data = savestate.capture(session)
    for attr, key in savestate.SAVED.items():
        top = key.split(".")[0].split(" ")[0]
        assert top in data, f"{attr}: {key} is not in the state"


def test_every_score_field_is_saved_or_derived():
    score = Score()
    assert set(score.to_state()) | DERIVED == set(score.__dataclass_fields__)


def test_every_action_is_saved_by_the_autosave_unless_listed():
    assert NO_AUTOSAVE <= {a.id for a in ACTIONS}


# --- save and restore -----------------------------------------------------------------


def test_save_state_action_writes_the_files(tmp_path, named):
    session, frame = games(tmp_path, named)
    session.dispatch("reference.take")
    session.dispatch("state.save")
    folders = [p for p in (tmp_path / "saves").iterdir() if not p.name.startswith("autosave")]
    assert len(folders) == 1 and folders[0].name.endswith("_turn2")
    names = {p.name for p in folders[0].iterdir()}
    assert names == {"state.json", "frame.png", "empty_board.png"}
    assert json.loads((folders[0] / "state.json").read_text())["files"]["reference"] == "empty_board.png"
    session.dispatch("state.save")  # a second save in the same minute does not replace the first
    assert len([p for p in (tmp_path / "saves").iterdir() if not p.name.startswith("autosave")]) == 2
    assert session.state()["notice"]["level"] == "done"


def test_restore_gives_the_same_state(tmp_path, named):
    session, frame = games(tmp_path, named)
    session.dispatch("state.save")
    folder = next(p for p in (tmp_path / "saves").iterdir() if not p.name.startswith("autosave"))
    before = savestate.capture(session)
    again = fresh(tmp_path, named)
    assert again.load_state(folder)
    after = savestate.capture(again)
    for data in (before, after):
        data.pop("saved_at")
        data.pop("files")  # the restored session has drawn no picture yet
    assert after == before
    assert again.ui_layout == "inspector" and again.turn == 2 and again.map_layout == "Small"
    state = again.state()
    assert state["mode"] == "snapshot" and state["naming"] and state["grid"] is False
    assert state["selected_wounds"]["current"] == 4 and state["ring_in"] == 3
    assert state["scores"][0]["cp"] == 2 and state["scores"][1]["primary"] == "tac"
    assert [b["label"] for b in state["bases"]] == [b["label"] for b in session.state()["bases"]]
    assert np.array_equal(again.snap, cv2.imread(str(folder / "frame.png")))


def test_restore_keeps_a_removed_base_and_the_openings_and_the_reference(tmp_path, named):
    session, frame = games(tmp_path, named)
    session.dispatch("reference.take")
    count = len(session.state()["bases"])
    session.dispatch("base.select", index=0)
    session.dispatch("base.remove")
    assert len(session.state()["bases"]) == count - 1
    session.dispatch("state.save")
    folder = next(p for p in (tmp_path / "saves").iterdir() if not p.name.startswith("autosave"))
    again = fresh(tmp_path, named)
    assert again.load_state(folder)
    assert len(again.state()["bases"]) == count - 1 and again.state()["reference_set"]
    assert np.array_equal(again.reference_frame, session.reference_frame)


@pytest.mark.parametrize("damage", ["missing", "not json", "other schema", "bad value"])
def test_a_bad_save_gives_a_notice_and_false(tmp_path, named, damage):
    session, _ = games(tmp_path, named)
    session.dispatch("state.save")
    folder = next(p for p in (tmp_path / "saves").iterdir() if not p.name.startswith("autosave"))
    file = folder / "state.json"
    data = json.loads(file.read_text())
    if damage == "missing":
        file.unlink()
    elif damage == "not json":
        file.write_text("{")
    elif damage == "other schema":
        file.write_text(json.dumps({**data, "schema": 99}))
    else:
        file.write_text(json.dumps({**data, "scores": [{"cp": "x", "bogus": 1}, data["scores"][1]]}))
    again = fresh(tmp_path, named)
    assert not again.load_state(folder)
    assert again.state()["notice"]["level"] == "error" and "not loaded" in again.state()["notice"]["text"]


def test_a_state_without_snapshot_or_reference_restores(tmp_path):
    cfg = ident_cfg()
    frame, _ = render_board(cfg)
    session = make_session(tmp_path, cfg, frame)
    settled(session, frame)
    session.dispatch("game.turn", delta=2)
    session.dispatch("state.save")
    folder = next(p for p in (tmp_path / "saves").iterdir() if not p.name.startswith("autosave"))
    assert {p.name for p in folder.iterdir()} <= {"state.json"}
    again = make_session(tmp_path / "again", cfg, frame)
    assert again.load_state(folder) and again.turn == 3 and again.state()["mode"] == "live"


# --- the auto-save --------------------------------------------------------------------


def test_autosave_writes_after_an_action_and_keeps_one_folder_per_run(tmp_path, named):
    cfg, frame = named
    session = make_session(tmp_path, cfg, frame, teams=("orks",))
    settled(session, frame)
    assert not (tmp_path / "saves").exists()  # nothing before the first action
    session.dispatch("game.turn", delta=1)
    folder = session.autosave_path
    assert folder.parent == tmp_path / "saves" and folder.name.startswith("autosave-")
    assert json.loads((folder / "state.json").read_text())["turn"] == 2
    session.dispatch("game.turn", delta=1)
    assert json.loads((folder / "state.json").read_text())["turn"] == 3
    assert [p.name for p in (tmp_path / "saves").iterdir()] == [folder.name]


def test_autosave_skips_the_pointer_and_the_save_action(tmp_path, named):
    cfg, frame = named
    session = make_session(tmp_path, cfg, frame)
    settled(session, frame)
    session.dispatch("board.hover", x=10.0, y=10.0)
    assert not session.autosave_path.exists()


def test_autosave_does_not_rewrite_the_frame_when_it_is_the_same(tmp_path, named):
    session, frame = games(tmp_path, named)
    frame_file = session.autosave_path / "frame.png"
    stamp = frame_file.stat().st_mtime_ns
    session.dispatch("game.turn", delta=1)
    assert frame_file.stat().st_mtime_ns == stamp


def test_a_new_run_makes_a_new_autosave_folder(tmp_path, named, monkeypatch):
    cfg, frame = named
    times = iter(["20260101-000001", "20260101-000002"])
    monkeypatch.setattr("battlesight.live.time.strftime", lambda fmt, *a: next(times) if fmt.startswith("autosave") else "x")
    one = make_session(tmp_path, cfg, frame)
    two = make_session(tmp_path, cfg, frame)
    assert one.autosave_path != two.autosave_path


def test_autosave_problem_is_a_notice_not_an_exception(tmp_path, named):
    cfg, frame = named
    (tmp_path / "saves").write_text("a file, not a folder")
    session = make_session(tmp_path, cfg, frame)
    settled(session, frame)
    session.dispatch("game.turn", delta=1)
    assert session.turn == 2 and "auto-save failed" in session.state()["notice"]["text"]


def test_autosave_can_be_off(tmp_path, named):
    cfg, frame = named
    session = make_session(tmp_path, cfg, frame, autosave=False)
    session.dispatch("game.turn", delta=1)
    assert not (tmp_path / "saves").exists()


def test_the_default_saves_folder_is_not_data_saves_in_tests(tmp_path, named):
    from battlesight import live

    assert live.SAVES_DIR.parent == tmp_path


# --- --state --------------------------------------------------------------------------


def test_cli_state_restores_and_a_bad_path_exits_2(tmp_path, named, monkeypatch, capsys):
    import battlesight.app as app

    session, frame = games(tmp_path, named)
    session.dispatch("state.save")
    folder = next(p for p in (tmp_path / "saves").iterdir() if not p.name.startswith("autosave"))
    cfg, frame = named
    img = tmp_path / "board.png"
    cv2.imwrite(str(img), frame)
    seen = {}

    def drive(live_session, camera):
        seen["turn"], seen["layout"] = live_session.turn, live_session.ui_layout
        return 0

    from battlesight.ui import shell

    monkeypatch.setattr(shell, "run_window", drive)
    assert app.main(["--image-source", str(img), "--state", str(folder), "--gallery-root", str(tmp_path / "gallery"), "--team", "orks"]) == 0
    assert seen == {"turn": 2, "layout": "inspector"}
    assert app.main(["--image-source", str(img), "--state", str(tmp_path / "nothing"), "--gallery-root", str(tmp_path / "gallery"), "--team", "orks"]) == 2
