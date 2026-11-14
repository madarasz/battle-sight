"""Game core of the live app (`live.LiveSession`) on synthetic boards. No camera,
no window, no model download."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import battlesight.app as app
import battlesight.live as live_mod
from battlesight import ident as idt
from battlesight import overlay, walls
from battlesight.calib import TagDetector
from battlesight.camera import CameraSettings, ImageSource
from battlesight.live import ACTIONS, PREVIEW_PAD_MM, LiveSession

from .conftest import render_board
from .test_ident import SIZES, enroll_from, ident_cfg, make_gallery, paint_board

TRAIN = {"Nob": ((150.0, 150.0), 0.0), "Grot": ((300.0, 200.0), 40.0), "Burna": ((450.0, 500.0), 90.0)}
GAME = {"Nob": ((300.0, 350.0), 120.0), "Grot": ((150.0, 500.0), 200.0), "Burna": ((450.0, 200.0), 300.0)}
SETTLE = 7  # more than settle_frames of the test config


def make_session(tmp_path, cfg, frame, **over) -> LiveSession:
    size = (frame.shape[1], frame.shape[0])
    args = dict(gallery_root=tmp_path / "gallery", reference_path=tmp_path / "ref" / "empty.png", snapshot_dir=tmp_path / "snaps", maps_root=tmp_path / "maps")
    args.update(over)
    return LiveSession(cfg, CameraSettings(), size, **args)


def settled(session: LiveSession, frame):
    for _ in range(SETTLE):
        live = session.step(frame)
    return live


@pytest.fixture
def named(tmp_path):
    """Gallery with three operatives, and a game frame with them moved and turned."""
    cfg = ident_cfg()
    gallery = make_gallery(tmp_path, cfg)
    enroll_from(gallery, cfg, paint_board(cfg, TRAIN), {n: c for n, (c, _) in TRAIN.items()})
    return cfg, paint_board(cfg, GAME)


def assert_json_safe(state: dict) -> None:
    json.dumps(state, allow_nan=False)


# --- contract -------------------------------------------------------------------


def test_actions_are_unique_and_core_actions_have_handlers(tmp_path):
    cfg = ident_cfg()
    frame, _ = render_board(cfg)
    session = make_session(tmp_path, cfg, frame)
    ids = [a.id for a in ACTIONS]
    keys = [a.key for a in ACTIONS if a.key]
    assert len(ids) == len(set(ids)) and len(keys) == len(set(keys))
    assert {a.id for a in ACTIONS if not a.shell} == set(session._handlers)
    state = settled(session, frame).state
    for a in ACTIONS:
        assert a.enabled(state) in (True, False)
        assert a.checked is None or a.checked(state) in (True, False)
    with pytest.raises(KeyError):
        session.dispatch("app.quit")


def test_live_state(tmp_path):
    cfg = ident_cfg()
    frame, _ = render_board(cfg)
    session = make_session(tmp_path, cfg, frame)
    live = settled(session, frame)
    s = live.state
    assert_json_safe(s)
    assert s["mode"] == "live" and s["camera"] and s["drift"]["ok"] and s["drift"]["status"] == "OK"
    assert s["lens"] == "nominal" and s["reference_set"] is False and s["snapshot_ok"] is None
    assert s["resolution"] == [frame.shape[1], frame.shape[0]]
    small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    assert session.step(small).state["resolution"] == [small.shape[1], small.shape[0]]
    assert s["bases"] == [] and s["selected"] is None and s["notice"] is None
    assert s["debug"]["tags"] == 4 and s["debug"]["fit_max_mm"] is not None
    assert live.image is not frame


def test_debug_values_use_the_clock_and_reset_when_the_camera_is_down(tmp_path):
    cfg = ident_cfg()
    frame, _ = render_board(cfg)
    now = [0.0]
    session = make_session(tmp_path, cfg, frame, clock=lambda: now[0])
    debug = session.state()["debug"]
    assert debug["fps"] is None and debug["detect_ms"] is None
    session.step(frame)
    assert session.state()["debug"]["fps"] is None  # one frame has no interval
    now[0] += 0.1
    session.step(frame)
    assert session.state()["debug"]["fps"] == pytest.approx(10.0)
    assert session.state()["debug"]["detect_ms"] == 0.0
    now[0] += 0.05
    session.step(frame)
    assert session.state()["debug"]["fps"] == pytest.approx(11.0)  # smoothed
    assert session.step(None).state["debug"]["fps"] is None
    json.dumps(session.state())


def test_crop_of_the_selected_base_is_next_to_the_state(tmp_path, named):
    cfg, frame = named
    session = make_session(tmp_path, cfg, frame, teams=("orks",))
    settled(session, frame)
    session.dispatch("snapshot.name")
    live = session.step(frame)
    assert live.crop is None
    x, y = live.state["bases"][1]["position_px"]
    session.dispatch("board.click", x=x, y=y)
    live = session.step(frame)
    full = session.session.crops[1]
    assert live.crop.flags.c_contiguous and live.crop.shape[0] < full.shape[0]
    diameter = session.session.diameters_mm[1]
    side_mm = live.crop.shape[0] / (full.shape[0] / (diameter + 2 * idt.STORE_PAD_MM))
    assert side_mm == pytest.approx(diameter + 2 * PREVIEW_PAD_MM, abs=1.0)
    session.dispatch("base.deselect")
    assert session.current().crop is None


def test_view_grid_actions_change_state_and_image(tmp_path):
    cfg = ident_cfg()
    frame, _ = render_board(cfg)
    session = make_session(tmp_path, cfg, frame)
    session.dispatch("view.top_down")  # the camera view
    camera = settled(session, frame)

    session.dispatch("view.grid")
    no_grid = session.step(frame)
    assert no_grid.state["grid"] is False and not np.array_equal(no_grid.image, camera.image)

    session.dispatch("view.top_down")
    top = session.step(frame)
    assert top.state["top_down"] is True and top.image.shape != camera.image.shape


def test_live_view_opens_with_adjusted_perspective(tmp_path):
    cfg = ident_cfg()
    frame, _ = render_board(cfg)
    session = make_session(tmp_path, cfg, frame)
    assert next(a for a in ACTIONS if a.id == "view.top_down").label == "Adjust perspective"
    live = settled(session, frame)
    assert live.state["top_down"] is True and live.image.shape != frame.shape


def _cover_tags(cfg, frame, ids):
    """The frame with a grey patch on each of these tags (a hand over a card)."""
    out = frame.copy()
    for tag in (t for i, t in TagDetector(cfg).detect(frame).items() if i in ids):
        (x, y), r = tag.center, int(tag.side_px)
        cv2.rectangle(out, (int(x) - r, int(y) - r), (int(x) + r, int(y) + r), (128, 128, 128), -1)
    return out


@pytest.mark.parametrize("ids", [(1,), (0, 2)])
def test_adjusted_perspective_stays_when_one_or_two_tags_are_covered(tmp_path, ids):
    cfg = ident_cfg()
    frame, _ = render_board(cfg)
    session = make_session(tmp_path, cfg, frame)
    full = settled(session, frame)
    covered = settled(session, _cover_tags(cfg, frame, ids))
    assert covered.state["drift"]["status"] == "LOST" and covered.state["drift"]["lost"] == list(ids)
    assert covered.state["drift"]["text"].startswith("TAG LOST")
    assert covered.image.shape == full.image.shape != frame.shape  # still top-down
    assert covered.state["debug"]["fit_max_mm"] == full.state["debug"]["fit_max_mm"]  # the saved position
    back = session.step(frame)
    assert back.state["drift"]["status"] == "OK" and back.image.shape == full.image.shape


def test_board_tracker_holds_the_view_while_two_tags_are_covered():
    from battlesight.live import BoardTracker

    cfg = ident_cfg()
    frame, _ = render_board(cfg)
    board = BoardTracker(cfg, CameraSettings())
    assert board.view is None and board.lens is None
    board.update(frame)
    assert board.view is not None and board.lens is not None  # all 4 tags: a fit before the lock
    for _ in range(SETTLE):
        board.update(frame)
    locked = board.view.H.copy()
    covered = _cover_tags(cfg, frame, (1, 3))
    for _ in range(SETTLE):
        report = board.update(covered)
        assert len(board.obs) == 2 and np.array_equal(board.view.H, locked)  # the saved position
    assert report.held
    three = _cover_tags(cfg, frame, (0, 1, 3))
    for _ in range(SETTLE):
        board.update(three)
    assert board.view is None


def test_adjusted_perspective_stops_when_three_tags_are_covered_or_the_board_moved(tmp_path):
    cfg = ident_cfg()
    frame, _ = render_board(cfg)
    session = make_session(tmp_path, cfg, frame)
    settled(session, frame)
    three = settled(session, _cover_tags(cfg, frame, (0, 1, 2)))
    assert three.state["drift"]["status"] == "LOST" and three.image.shape == frame.shape
    settled(session, frame)
    shifted, _ = render_board(cfg, shift_px=(8.0, 0.0))  # 4 mm
    moved = settled(session, _cover_tags(cfg, shifted, (1,)))
    assert moved.state["drift"]["status"] == "LOST" and moved.image.shape == frame.shape


def test_camera_down_keeps_working(tmp_path):
    cfg = ident_cfg()
    frame, _ = render_board(cfg)
    session = make_session(tmp_path, cfg, frame)
    first = session.step(None)
    assert first.image is None and first.state["camera"] is False and first.state["drift"] is None
    assert_json_safe(first.state)
    session.dispatch("snapshot.name")
    assert session.state()["notice"]["level"] == "error"
    assert session.step(frame).state["camera"] is True
    assert session.step(None).image is not None


# --- snapshot -------------------------------------------------------------------


def test_bases_snapshot_and_back(tmp_path):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    session = make_session(tmp_path, cfg, frame)
    settled(session, frame)
    session.dispatch("snapshot.bases")
    s = session.step(frame).state
    assert_json_safe(s)
    assert s["mode"] == "snapshot" and s["snapshot_ok"] and not s["naming"] and not s["busy"]
    assert len(s["bases"]) == 3 and all(b["label"] is None for b in s["bases"])
    x, y = s["bases"][0]["position_px"]
    session.dispatch("board.select", x=x, y=y)
    assert session.state()["notice"]["text"] == "push space to name the operatives"
    session.dispatch("snapshot.back")  # the selection
    session.dispatch("snapshot.back")
    assert session.step(frame).state["mode"] == "live"


def test_failed_snapshot_says_why(tmp_path):
    cfg = ident_cfg()
    frame, mm2px = render_board(cfg)
    p = mm2px((-25, -25)).astype(int)
    covered = frame.copy()
    cv2.rectangle(covered, tuple(p - 60), tuple(p + 60), (160, 160, 160), -1)
    session = make_session(tmp_path, cfg, covered)
    session.step(covered)
    session.dispatch("snapshot.bases")
    s = session.step(covered).state
    assert s["mode"] == "snapshot" and s["snapshot_ok"] is False and s["bases"] == []
    assert s["notice"]["level"] == "error" and s["notice"]["text"].startswith("corners failed")


def test_no_gallery_teams_gives_message(tmp_path):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    session = make_session(tmp_path, cfg, frame)
    session.step(frame)
    session.dispatch("snapshot.name")
    s = session.state()
    assert not s["naming"] and len(s["bases"]) == 3
    assert "no gallery teams" in s["notice"]["text"]


@pytest.mark.parametrize("display_width", [1600, 500])
def test_name_fix_flow(tmp_path, named, display_width):
    cfg, frame = named
    session = make_session(tmp_path, cfg, frame, teams=("orks",), display_width=display_width)
    settled(session, frame)
    for action in ("names.accept", "names.undo"):
        session.dispatch(action)
        assert session.state()["notice"]["text"] == "push space first"

    session.dispatch("snapshot.name")
    live = session.step(frame)
    s = live.state
    assert_json_safe(s)
    assert s["naming"] and s["teams"] == ["orks"] and len(s["bases"]) == 3 and s["unconfirmed"] > 0
    assert live.image.shape[1] <= display_width

    # Click the base at the position that the state gives: the mapping back to mm is right.
    base = s["bases"][1]
    session.dispatch("board.click", x=base["position_px"][0], y=base["position_px"][1])
    s = session.step(frame).state
    assert_json_safe(s)
    assert s["selected"] == 1 and s["candidates"]
    label = s["candidates"][0]["label"]

    session.dispatch("base.choose", label=label)
    s = session.step(frame).state
    assert s["selected"] is None and s["can_undo"]
    assert s["bases"][1]["label"] == label and s["bases"][1]["confirmed"]

    session.dispatch("names.undo")
    s = session.step(frame).state
    assert not s["can_undo"] and not s["bases"][1]["confirmed"]

    session.dispatch("names.accept")
    s = session.step(frame).state
    assert s["unconfirmed"] == 0


def test_remove_base_drops_a_false_base(tmp_path, named):
    cfg, frame = named
    session = make_session(tmp_path, cfg, frame, teams=("orks",))
    settled(session, frame)
    session.dispatch("base.remove")
    assert session.state()["notice"]["text"] == "click a base first"
    session.dispatch("snapshot.name")
    s = session.step(frame).state
    kept = [b["label"] for k, b in enumerate(s["bases"]) if k != 1]
    session.dispatch("base.select", index=1)
    session.dispatch("ring.3")
    session.dispatch("base.remove")
    s = session.step(frame).state
    assert_json_safe(s)
    assert len(s["bases"]) == 2 and [b["label"] for b in s["bases"]] == kept
    assert s["selected"] is None and s["ring_in"] is None and s["notice"]["text"] == "base removed"
    assert [b["index"] for b in s["bases"]] == [0, 1]
    session.dispatch("base.select", index=1)  # the measurement uses the bases that are left
    session.dispatch("ring.3")
    session.step(frame)


def test_remove_base_without_names(tmp_path, named):
    cfg, frame = named
    session = make_session(tmp_path, cfg, frame, teams=("orks",))
    settled(session, frame)
    session.dispatch("snapshot.bases")
    session.dispatch("base.select", index=0)
    session.dispatch("base.remove")
    s = session.step(frame).state
    assert len(s["bases"]) == 2 and not s["naming"]


def test_add_operative_finds_a_removed_base_and_opens_its_name_list(tmp_path, named):
    cfg, frame = named
    session = make_session(tmp_path, cfg, frame, teams=("orks",))
    settled(session, frame)
    session.dispatch("snapshot.name")
    s = session.step(frame).state
    assert next(a for a in ACTIONS if a.id == "base.add").enabled(s)
    gone = s["bases"][1]
    session.dispatch("base.select", index=1)
    session.dispatch("base.remove")
    assert len(session.state()["bases"]) == 2

    session.dispatch("base.add")
    s = session.step(frame).state
    assert s["adding"] and s["selected"] is None
    assert session.current().crop is None  # no pointer yet
    px, py = gone["position_px"]
    session.dispatch("board.hover", x=px, y=py)
    aim = session.current().crop
    assert aim is not None and aim.shape[0] == aim.shape[1] == round((live_mod.AIM_DIAMETER_MM + 2 * PREVIEW_PAD_MM) * cfg.bases_px_per_mm)
    session.dispatch("board.hover", x=None, y=None)
    assert session.current().crop is None
    session.dispatch("board.click", x=5.0, y=5.0)  # no base there: still waiting
    s = session.step(frame).state
    assert s["adding"] and "no base found" in s["notice"]["text"]
    x, y = gone["position_px"]
    session.dispatch("board.click", x=x + 4.0, y=y - 3.0)  # near the centre
    s = session.step(frame).state
    assert_json_safe(s)
    assert not s["adding"] and len(s["bases"]) == 3
    assert s["selected"] == 2 and s["new_base"] == 2 and s["candidates"]
    assert s["bases"][2]["diameter_mm"] == gone["diameter_mm"]
    session.dispatch("board.click", x=x, y=y)  # the same base again: a click selects (and clears) it
    assert len(session.state()["bases"]) == 3
    session.dispatch("base.select", index=2)
    assert session.state()["new_base"] is None

    session.dispatch("base.select", index=2)
    label = session.state()["candidates"][0]["label"]
    session.dispatch("base.choose", label=label)
    s = session.step(frame).state
    assert s["bases"][2]["label"] == label and s["bases"][2]["confirmed"] and s["new_base"] is None
    session.dispatch("names.undo")
    assert session.state()["bases"][2]["label"] is None or not session.state()["bases"][2]["confirmed"]


def test_add_operative_before_identification_starts_an_empty_snapshot(tmp_path, named):
    cfg, frame = named
    session = make_session(tmp_path, cfg, frame, teams=("orks",))
    live = settled(session, frame)
    assert next(a for a in ACTIONS if a.id == "base.add").enabled(live.state)
    session.dispatch("snapshot.bases")  # positions of the bases, for the clicks
    where = session.step(frame).state["bases"]
    session.dispatch("snapshot.back")
    session._close_snapshot()  # no snapshot, no names: as at the start
    session.dispatch("base.add")
    s = session.step(frame).state
    assert_json_safe(s)
    assert s["adding"] and s["mode"] == "snapshot" and s["naming"] and s["bases"] == []
    x, y = where[0]["position_px"]
    session.dispatch("board.click", x=x, y=y)
    s = session.step(frame).state
    assert len(s["bases"]) == 1 and s["selected"] == 0 and s["new_base"] == 0 and s["candidates"]
    session.dispatch("base.choose", label=s["candidates"][0]["label"])
    session.dispatch("base.add")  # a second operative in the same snapshot
    x, y = where[1]["position_px"]
    session.dispatch("board.click", x=x, y=y)
    s = session.step(frame).state
    assert len(s["bases"]) == 2 and s["bases"][0]["confirmed"]


def test_add_operative_after_bases_only_and_from_the_live_video(tmp_path, named):
    cfg, frame = named
    session = make_session(tmp_path, cfg, frame, teams=("orks",))
    settled(session, frame)
    session.dispatch("snapshot.bases")
    session.dispatch("base.add")  # bases only: the bases get a name session, all unknown
    s = session.step(frame).state
    assert s["adding"] and s["naming"] and len(s["bases"]) == 3 and all(b["label"] is None for b in s["bases"])
    session.dispatch("snapshot.back")  # stops adding
    session.dispatch("snapshot.back")  # back to the live video, the names stay
    assert session.state()["mode"] == "live"
    session.dispatch("base.add")  # the snapshot of those names comes back
    s = session.step(frame).state
    assert s["adding"] and s["mode"] == "snapshot" and len(s["bases"]) == 3


def test_add_operative_does_not_double_an_existing_base_and_only_esc_cancels(tmp_path, named):
    cfg, frame = named
    session = make_session(tmp_path, cfg, frame, teams=("orks",))
    settled(session, frame)
    session.dispatch("snapshot.name")
    s = session.step(frame).state
    session.dispatch("base.add")
    x, y = s["bases"][0]["position_px"]
    session.dispatch("board.click", x=x, y=y)  # on a base that is known
    s = session.step(frame).state
    assert s["adding"] and len(s["bases"]) == 3
    session.dispatch("snapshot.back")  # stops adding, stays in the snapshot
    s = session.step(frame).state
    assert not s["adding"] and s["mode"] == "snapshot"
    session.dispatch("base.add")
    session.dispatch("base.add")  # a second push changes nothing: only Esc stops it
    assert session.state()["adding"]
    session.dispatch("snapshot.back")
    assert not session.state()["adding"] and session.state()["mode"] == "snapshot"


def test_back_goes_live_and_keeps_names_and_selection(tmp_path, named):
    cfg, frame = named
    session = make_session(tmp_path, cfg, frame, teams=("orks",))
    settled(session, frame)
    session.dispatch("snapshot.name")
    snap = session.step(frame)
    base = snap.state["bases"][0]
    session.dispatch("board.click", x=base["position_px"][0], y=base["position_px"][1])
    assert session.state()["selected"] == 0
    session.dispatch("snapshot.back")
    live = session.step(frame)
    s = live.state
    assert s["mode"] == "live" and s["selected"] == 0 and s["naming"]
    assert [b["label"] for b in s["bases"]] == [b["label"] for b in snap.state["bases"]]
    assert s["bases"][0]["position_px"] == pytest.approx(base["position_px"], abs=1.0)  # same top-down mapping
    assert live.image is not snap.image
    # Clicks work on the live video too.
    other = s["bases"][1]
    session.dispatch("board.select", x=other["position_px"][0], y=other["position_px"][1])
    assert session.state()["target"] == 1
    session.dispatch("snapshot.back")  # in the live video: clears the selection
    s = session.state()
    assert s["mode"] == "live" and s["selected"] is None and s["naming"]


def test_live_camera_view_without_a_board_fit_has_no_base_positions(tmp_path, named):
    cfg, frame = named
    session = make_session(tmp_path, cfg, frame, teams=("orks",))
    settled(session, frame)
    session.dispatch("snapshot.name")
    session.step(frame)
    session.dispatch("snapshot.back")
    session.dispatch("view.top_down")
    camera = session.step(frame).state
    assert camera["top_down"] is False and all(b["position_px"] is not None for b in camera["bases"])
    blank = np.full_like(frame, 128)
    for _ in range(SETTLE):  # the drift monitor waits some frames before LOST
        lost = session.step(blank).state
    assert lost["drift"]["status"] == "LOST"
    assert len(lost["bases"]) == 3 and all(b["position_px"] is None for b in lost["bases"])
    assert_json_safe(lost)


def test_base_select_by_index(tmp_path, named):
    cfg, frame = named
    session = make_session(tmp_path, cfg, frame, teams=("orks",))
    settled(session, frame)
    session.dispatch("base.select", index=0)
    assert session.state()["selected"] is None and session.state()["notice"]["level"] == "error"
    session.dispatch("snapshot.name")
    session.step(frame)
    session.dispatch("base.select", index=2)
    session.dispatch("ring.3")
    session.dispatch("base.select", index=1)
    s = session.state()
    assert s["selected"] == 1 and s["ring_in"] is None and s["candidates"]


def test_team_lists_follow_the_snapshot(tmp_path, named):
    cfg, frame = named
    session = make_session(tmp_path, cfg, frame, teams=("orks",))
    s = session.state()
    assert s["team_choices"] == ["orks"] and s["team_slots"] == ["orks", None]
    roster = s["rosters"][0]
    assert s["rosters"][1] is None and roster["team"] == "orks"
    assert {r["name"] for r in roster["operatives"]} == set(GAME)
    assert all(r["base_index"] is None and r["score"] is None and not r["confirmed"] for r in roster["operatives"])
    settled(session, frame)
    session.dispatch("snapshot.name")
    s = session.step(frame).state
    assert_json_safe(s)
    rows = {r["name"]: r for r in s["rosters"][0]["operatives"]}
    for b in s["bases"]:
        row = rows[b["label"].split("/")[1]]
        assert row["base_index"] == b["index"] and row["score"] == b["score"]
    session.dispatch("names.accept")
    assert all(r["confirmed"] for r in session.state()["rosters"][0]["operatives"])


def test_team_change_resets_the_gallery_and_keeps_the_names(tmp_path, named):
    cfg, frame = named
    session = make_session(tmp_path, cfg, frame)
    assert session.state()["team_slots"] == [None, None]
    settled(session, frame)
    session.dispatch("snapshot.name")  # no team chosen: all gallery teams
    assert session.step(frame).state["naming"]
    assert session.gallery is not None
    session.dispatch("team.set", slot=1, name="orks")
    s = session.state()
    assert s["team_slots"] == [None, "orks"] and session.teams == ["orks"]
    assert session.gallery is None and not session.gallery_tried and s["naming"]
    assert "identify" in s["notice"]["text"]
    session.dispatch("team.set", slot=0, name="orks")
    s = session.state()
    assert s["team_slots"] == [None, "orks"] and "other side" in s["notice"]["text"]
    session.dispatch("team.set", slot=0, name="elves")
    assert "no team" in session.state()["notice"]["text"]
    session.dispatch("team.set", slot=1, name=None)
    assert session.state()["team_slots"] == [None, None] and session.teams is None


def test_click_with_no_operative_of_that_size(tmp_path):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    session = make_session(tmp_path, cfg, frame, teams=("orks",))
    idt.Gallery.for_teams(("orks",), cfg, tmp_path / "gallery").ensure_operative("orks", "Big", 50)
    session.pipe.set_roster(cfg.bases_diameters_mm)  # the team sizes alone (50 mm) would hide the other bases
    settled(session, frame)
    session.dispatch("snapshot.name")
    base = session.step(frame).state["bases"][0]
    session.dispatch("board.select", x=base["position_px"][0], y=base["position_px"][1])
    s = session.state()
    assert s["selected"] == 0 and s["candidates"] == [] and "mm base in the gallery" in s["notice"]["text"]


# --- reference and saved frames ---------------------------------------------------


def test_reference_take_clear_and_bad_frame(tmp_path):
    cfg = ident_cfg()
    frame, _ = render_board(cfg)
    path = tmp_path / "ref" / "empty.png"
    session = make_session(tmp_path, cfg, frame)
    session.step(frame)
    session.dispatch("reference.take")
    s = session.state()
    assert s["reference_set"] and s["notice"]["level"] == "done" and path.is_file()

    assert make_session(tmp_path, cfg, frame).state()["reference_set"]  # loaded at start

    session.dispatch("reference.clear")
    assert session.state()["reference_set"] is False

    blank = np.full_like(frame, 128)
    session.step(blank)
    session.dispatch("reference.take")
    s = session.state()
    assert s["reference_set"] is False and s["notice"]["level"] == "error"


def test_save_frame(tmp_path):
    cfg = ident_cfg()
    frame, _ = render_board(cfg)
    session = make_session(tmp_path, cfg, frame)
    session.step(frame)
    session.dispatch("frame.save")
    assert len(list((tmp_path / "snaps").glob("snap_*.png"))) == 1
    assert session.state()["notice"]["level"] == "done"


# --- --image-source -----------------------------------------------------------------


def test_image_source_gives_the_same_frame(tmp_path):
    frame = np.full((40, 60, 3), 90, np.uint8)
    path = tmp_path / "shot.png"
    cv2.imwrite(str(path), frame)
    source = ImageSource(path, fps=1000)
    assert source.open() and source.size == (60, 40)
    assert np.array_equal(source.read(), frame) and np.array_equal(source.read(), frame)
    source.release()
    assert source.read() is None


def test_image_source_bad_file(tmp_path, capsys):
    (tmp_path / "text.png").write_text("not an image")
    assert not ImageSource(tmp_path / "missing.png").open()
    assert not ImageSource(tmp_path / "text.png").open()
    assert app.main(["--image-source", str(tmp_path / "missing.png")]) == 2
    assert "cannot read image" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["--image-source", "a.png", "--source", "0"], ["enroll", "--image-source", "a.png", "--team", "t", "--operative", "o"]])
def test_image_source_bad_options(argv):
    with pytest.raises(SystemExit) as exc:
        app.main(argv)
    assert exc.value.code == 2


def test_image_source_is_not_an_option_of_a_command(tmp_path):
    with pytest.raises(SystemExit) as exc:
        app.main(["--image-source", "a.png", "gallery", "--team", "t"])
    assert exc.value.code == 2


# --- the v key over a snapshot ------------------------------------------------------


def test_v_changes_the_snapshot_view_and_clicks_still_hit(tmp_path, named):
    cfg, frame = named
    session = make_session(tmp_path, cfg, frame, teams=("orks",))
    settled(session, frame)
    session.dispatch("view.top_down")  # the camera view
    assert session.state()["top_down"] is False

    session.dispatch("snapshot.name")
    top = session.step(frame)
    assert top.state["top_down"] is True  # opens top-down from the camera view

    session.dispatch("view.top_down")
    camera = session.step(frame)
    assert camera.state["top_down"] is False and camera.state["mode"] == "snapshot"
    assert camera.image.shape != top.image.shape
    base = camera.state["bases"][2]
    assert base["position_px"] != top.state["bases"][2]["position_px"]
    session.dispatch("board.click", x=base["position_px"][0], y=base["position_px"][1])
    assert session.state()["selected"] == 2

    session.dispatch("snapshot.back")  # clear the selection
    session.dispatch("snapshot.back")  # close the snapshot
    assert session.step(frame).state["top_down"] is False  # the live view kept its own view

    session.dispatch("snapshot.bases")
    assert session.step(frame).state["top_down"] is True  # the next snapshot opens top-down again


# --- rings and distances (priority 5) -----------------------------------------------


def bases_snapshot(tmp_path, cfg, frame):
    session = make_session(tmp_path, cfg, frame)
    settled(session, frame)
    session.dispatch("snapshot.bases")
    return session, session.step(frame).state


def click_base(session, state, index, action="board.click"):
    x, y = state["bases"][index]["position_px"]
    session.dispatch(action, x=x, y=y)


def game_name(session, index) -> str:
    """The name in GAME of a found base."""
    center = session._found()[index].center_mm
    return min(GAME, key=lambda n: np.linalg.norm(np.subtract(GAME[n][0], center)))


def test_ring_keys_need_a_selected_base(tmp_path):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    session, _ = bases_snapshot(tmp_path, cfg, frame)
    session.dispatch("ring.3")
    s = session.state()
    assert s["ring_in"] is None and s["notice"]["text"] == "click a base first"
    assert not next(a for a in ACTIONS if a.id == "ring.3").enabled(s)


@pytest.mark.parametrize("inches", [1, 6, 9])
def test_ring_is_drawn_at_base_radius_plus_inches(tmp_path, inches):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    session, state = bases_snapshot(tmp_path, cfg, frame)
    click_base(session, state, 0)
    before = session.step(frame).image.copy()
    session.dispatch(f"ring.{inches}")
    live = session.step(frame)
    assert live.state["ring_in"] == inches and live.state["selected"] == 0
    assert_json_safe(live.state)
    assert next(a for a in ACTIONS if a.id == f"ring.{inches}").checked(live.state)

    rows, cols = np.nonzero(np.any(live.image != before, axis=2))
    assert len(rows) > 100
    px = np.c_[cols, rows] / session.win.scale
    mm = px / cfg.rectify_px_per_mm - cfg.rectify_margin_mm
    base = session._found()[0]
    radii = np.linalg.norm(mm - base.center_mm, axis=1)
    assert np.median(radii) == pytest.approx(0.5 * base.diameter_mm + inches * 25.4, abs=1.0)

    session.dispatch("ring.clear")
    cleared = session.step(frame)
    assert cleared.state["ring_in"] is None and np.array_equal(cleared.image, before)


def test_second_base_gives_rim_to_rim_distance_within_2_mm(tmp_path):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    session, state = bases_snapshot(tmp_path, cfg, frame)
    click_base(session, state, 0)
    assert session.state()["distance"] is None
    before = session.step(frame).image.copy()
    click_base(session, state, 1, "board.select")
    live = session.step(frame)
    s = live.state
    assert_json_safe(s)
    assert s["selected"] == 0 and s["target"] == 1 and not np.array_equal(live.image, before)
    a, b = game_name(session, 0), game_name(session, 1)
    truth = np.linalg.norm(np.subtract(GAME[a][0], GAME[b][0])) - 0.5 * SIZES[a] - 0.5 * SIZES[b]
    assert abs(s["distance"]["mm"] - truth) < 2.0
    assert s["distance"]["in"] == pytest.approx(s["distance"]["mm"] / 25.4, abs=1e-3)


def test_clicks_move_and_clear_the_target_and_the_selection(tmp_path):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    session, state = bases_snapshot(tmp_path, cfg, frame)
    click_base(session, state, 0)
    click_base(session, state, 1, "board.select")
    click_base(session, state, 2, "board.select")
    assert session.state()["target"] == 2 and session.state()["right_click"] == "measure"
    click_base(session, state, 2, "board.select")
    s = session.state()
    assert s["target"] is None and s["selected"] == 0 and s["distance"] is None
    click_base(session, state, 1, "board.select")
    click_base(session, state, 2)  # a left click on another base selects it
    s = session.state()
    assert s["selected"] == 2 and s["target"] is None
    session.dispatch("ring.4")
    click_base(session, state, 2)
    s = session.state()
    assert s["selected"] is None and s["ring_in"] is None


def test_click_on_the_empty_board_clears_the_selection(tmp_path):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    session, state = bases_snapshot(tmp_path, cfg, frame)
    click_base(session, state, 0)
    session.dispatch("board.click", x=1, y=1)
    assert session.state()["selected"] is None


def test_right_click_starts_a_new_selection(tmp_path, named):
    cfg, frame = named
    session = make_session(tmp_path, cfg, frame, teams=("orks",))
    settled(session, frame)
    session.dispatch("snapshot.name")
    state = session.step(frame).state
    click_base(session, state, 2, "board.select")  # nothing selected: select for names
    s = session.state()
    assert s["selected"] == 2 and s["right_click"] == "name" and s["candidates"]
    click_base(session, state, 1, "board.select")  # another base: the target
    session.dispatch("ring.2")
    s = session.state()
    assert s["selected"] == 2 and s["target"] == 1 and s["right_click"] == "measure"
    click_base(session, state, 2, "board.select")  # the selected base: names again, a new selection
    s = session.state()
    assert s["selected"] == 2 and s["target"] is None and s["ring_in"] is None and s["right_click"] == "name"


def test_view_change_keeps_ring_and_distance(tmp_path):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    session, state = bases_snapshot(tmp_path, cfg, frame)
    click_base(session, state, 0)
    click_base(session, state, 1, "board.select")
    session.dispatch("ring.2")
    top = session.step(frame)
    session.dispatch("view.top_down")
    camera = session.step(frame)
    assert camera.state["top_down"] is False and camera.image.shape != top.image.shape
    assert camera.state["ring_in"] == 2 and camera.state["distance"] == top.state["distance"]


def test_esc_keeps_the_measurement_and_a_new_snapshot_clears_it(tmp_path):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    session, state = bases_snapshot(tmp_path, cfg, frame)
    click_base(session, state, 0)
    click_base(session, state, 1, "board.select")
    session.dispatch("ring.2")
    before = session.state()
    session.dispatch("snapshot.back")
    s = session.state()
    assert s["mode"] == "live"
    assert (s["selected"], s["target"], s["ring_in"], s["distance"]) == (0, 1, 2, before["distance"])
    session.dispatch("snapshot.bases")
    s = session.state()
    assert s["selected"] is None and s["mode"] == "snapshot"


def test_choosing_a_name_clears_the_measurement(tmp_path, named):
    cfg, frame = named
    session = make_session(tmp_path, cfg, frame, teams=("orks",))
    settled(session, frame)
    session.dispatch("snapshot.name")
    state = session.step(frame).state
    click_base(session, state, 0)
    click_base(session, state, 1, "board.select")
    label = session.state()["candidates"][0]["label"]
    session.dispatch("base.choose", label=label, index=0)
    s = session.state()
    assert s["selected"] is None and s["target"] is None and s["bases"][0]["confirmed"]


SMALL_MAP = """schema = 1
name = "Small"
terrain = "Test World"
pillars = [[2, 2], [4, 2]]

[[wall]]
key = "A2"
from = [2, 2]
to = [4, 2]
opening = true
"""


def write_map(root, folder="test-world", file="small.toml", text=SMALL_MAP):
    path = root / folder / file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_map_layout_draws_terrain_and_toggles(tmp_path):
    cfg = ident_cfg()
    frame, _ = render_board(cfg)
    write_map(tmp_path / "maps")
    session = make_session(tmp_path, cfg, frame)
    plain = settled(session, frame)
    assert plain.state["terrain_types"] == ["Test World"] and plain.state["terrain_type"] == "Test World"
    assert plain.state["map_layouts"] == ["Small"] and plain.state["map_layout"] is None

    session.dispatch("map.layout", name="Small")
    shown = session.step(frame)
    assert shown.state["map_layout"] == "Small" and shown.state["terrain"] is True
    assert not np.array_equal(shown.image, plain.image)

    session.dispatch("view.terrain")
    hidden = session.step(frame)
    assert hidden.state["terrain"] is False and np.array_equal(hidden.image, plain.image)

    session.dispatch("map.layout", name=None)
    assert session.step(frame).state["map_layout"] is None


def test_light_terrain_shows_with_the_layout(tmp_path):
    cfg = ident_cfg()
    frame, _ = render_board(cfg)
    write_map(tmp_path / "maps", text='schema = 1\nname = "Small"\nterrain = "Test World"\n[[light]]\nkey = "C1"\nat = [3, 3]\n')
    session = make_session(tmp_path, cfg, frame)
    plain = settled(session, frame)
    session.dispatch("map.layout", name="Small")
    shown = session.step(frame)
    assert shown.state["map_layout"] == "Small" and not np.array_equal(shown.image, plain.image)
    session.dispatch("view.terrain")
    assert np.array_equal(session.step(frame).image, plain.image)


def test_bad_map_file_gives_a_notice_not_a_stop(tmp_path):
    cfg = ident_cfg()
    frame, _ = render_board(cfg)
    write_map(tmp_path / "maps", text=SMALL_MAP.replace("to = [4, 2]", "to = [4, 3]"))
    session = make_session(tmp_path, cfg, frame)
    session.dispatch("map.layout", name="Small")
    live = session.step(frame)
    assert live.state["map_layout"] is None and "horizontal or vertical" in live.state["notice"]["text"]
    session.dispatch("map.terrain_type", name="Nowhere")
    assert "Nowhere" in session.step(frame).state["notice"]["text"]


ZONE = '\n[[zone]]\nname = "grey"\nfrom = [0, 0]\nto = [4, 2]\n'


def test_drop_zones_are_off_until_turned_on(tmp_path):
    cfg = ident_cfg()
    frame, _ = render_board(cfg)
    write_map(tmp_path / "maps", text=SMALL_MAP + ZONE)
    session = make_session(tmp_path, cfg, frame)
    session.dispatch("map.layout", name="Small")
    walls = settled(session, frame)
    assert walls.state["zones"] is False
    session.dispatch("view.zones")
    zones = session.step(frame)
    assert zones.state["zones"] is True and not np.array_equal(zones.image, walls.image)
    session.dispatch("view.terrain")  # walls off, zones stay
    only_zones = session.step(frame)
    assert not np.array_equal(only_zones.image, zones.image)
    session.dispatch("view.zones")
    session.dispatch("map.layout", name=None)
    assert ACTIONS_BY_ID["view.zones"].enabled(session.step(frame).state) is False


ACTIONS_BY_ID = {a.id: a for a in ACTIONS}


def test_terrain_keys_are_off_until_turned_on(tmp_path):
    cfg = ident_cfg()
    frame, _ = render_board(cfg)
    write_map(tmp_path / "maps")
    session = make_session(tmp_path, cfg, frame)
    session.dispatch("map.layout", name="Small")
    plain = settled(session, frame)
    assert plain.state["keys"] is False and ACTIONS_BY_ID["view.keys"].enabled(plain.state)
    session.dispatch("view.keys")
    keyed = session.step(frame)
    assert keyed.state["keys"] is True and not np.array_equal(keyed.image, plain.image)


def opening_px(session) -> tuple[float, float]:
    center = session.openings[0].points.mean(axis=0)
    p = np.asarray(session._to_px(center.reshape(1, 2)), float).reshape(2) * session.win.scale
    return float(p[0]), float(p[1])


def test_click_selects_a_hatchway_in_live_and_snapshot(tmp_path):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    write_map(tmp_path / "maps")
    session = make_session(tmp_path, cfg, frame)
    session.dispatch("map.layout", name="Small")
    plain = settled(session, frame)
    assert len(session.openings) == 1
    x, y = opening_px(session)
    session.dispatch("board.click", x=x, y=y)
    s = session.step(frame)
    assert s.state["selected_opening"] == {"index": 0, "kind": session.openings[0].kind, "open": False}
    assert s.state["selected_opening"]["kind"] in ("hatchway", "breach")
    assert not np.array_equal(s.image, plain.image)  # outline drawn
    assert ACTIONS_BY_ID["base.deselect"].enabled(s.state)
    session.dispatch("board.click", x=x, y=y)  # the same opening again: clear
    assert session.state()["selected_opening"] is None

    session.dispatch("snapshot.bases")
    snap = session.step(frame).state
    x, y = opening_px(session)
    session.dispatch("board.select", x=x, y=y)
    assert session.state()["selected_opening"]["index"] == 0
    click_base(session, snap, 0)  # a base replaces the opening
    s = session.state()
    assert s["selected"] == 0 and s["selected_opening"] is None
    session.dispatch("board.click", x=x, y=y)
    s = session.state()
    assert s["selected"] is None and s["selected_opening"] is not None
    session.dispatch("base.deselect")
    assert session.state()["selected_opening"] is None
    session.dispatch("board.click", x=x, y=y)
    session.dispatch("map.layout", name=None)
    assert session.state()["selected_opening"] is None


def test_names_on_the_live_image_have_no_scores(tmp_path):
    from battlesight.overlay import draw_idents

    class C:
        center_mm = np.array([50.0, 50.0])
        diameter_mm = 32.0

    x = idt.Ident("orks/Nob", 0.5, 0.0)
    a = np.full((200, 200, 3), 128, np.uint8)
    b = a.copy()
    draw_idents(a, [C()], [x], lambda p: np.asarray(p, float), scores=True)
    draw_idents(b, [C()], [x], lambda p: np.asarray(p, float), scores=False)
    assert not np.array_equal(a, b)


def test_team_colors_paint_the_circles():
    from battlesight.overlay import BLUE, GREY, RED, draw_idents

    class C:
        def __init__(self, x):
            self.center_mm = np.array([x, 50.0])
            self.diameter_mm = 32.0

    bases = [C(40.0), C(100.0), C(160.0)]
    idents = [idt.Ident("orks/Nob", 0.9, 0.5, sure=True), idt.Ident("elves/Ari", 0.9, 0.5, sure=True), idt.Ident(None, 0.0, 0.0)]
    img = np.zeros((120, 220, 3), np.uint8)
    draw_idents(img, bases, idents, lambda p: np.asarray(p, float), team_colors={"orks": RED, "elves": BLUE})
    for x, color in ((40, RED), (100, BLUE), (160, GREY)):
        rim = img[30:51, x - 20 : x + 21].reshape(-1, 3).sum(axis=0).astype(float)  # upper half, no name there
        assert np.allclose(rim / rim.max(), np.array(color) / max(color), atol=0.05)  # anti-aliased: same hue
        name = img[60:120, x - 30 : x + 31].reshape(-1, 3)
        name = name[name.sum(axis=1) > 150].sum(axis=0).astype(float)  # the text, not its dark box
        assert np.allclose(name / name.max(), np.array(color) / max(color), atol=0.1)


def test_names_inside_put_the_name_in_the_middle_of_the_base():
    from battlesight.overlay import RED, draw_idents

    class C:
        center_mm = np.array([100.0, 100.0])
        diameter_mm = 32.0

    x = idt.Ident("orks/Nob", 0.9, 0.5, sure=True)
    img = np.zeros((200, 200, 3), np.uint8)
    draw_idents(img, [C()], [x], lambda p: np.asarray(p, float), scores=False, team_colors={"orks": RED}, inside=True)
    text = np.argwhere((img[:, :, 2] > 150) & (img[:, :, 0] < 80))  # red text, not the ring
    text = text[np.hypot(text[:, 0] - 100, text[:, 1] - 100) < 14]
    assert len(text) and abs(text[:, 0].mean() - 100) < 4  # rows centred on the base
    assert not img[118:140, 80:121].any()  # nothing under the base


# --- minimap ------------------------------------------------------------------------


def test_minimap_needs_a_layout_and_is_off_at_the_start(tmp_path):
    cfg = ident_cfg()
    frame, _ = render_board(cfg)
    write_map(tmp_path / "maps")
    session = make_session(tmp_path, cfg, frame)
    live = settled(session, frame)
    assert live.state["minimap"] is False and not ACTIONS_BY_ID["view.minimap"].enabled(live.state)
    session.dispatch("view.minimap")  # no layout: a hint, no change
    s = session.state()
    assert s["minimap"] is False and "layout" in s["notice"]["text"]
    session.dispatch("map.layout", name="Small")
    with_layout = session.step(frame)
    assert with_layout.state["minimap"] is False and ACTIONS_BY_ID["view.minimap"].enabled(with_layout.state)
    session.dispatch("view.minimap")
    mini = session.step(frame)
    assert mini.state["minimap"] is True and mini.state["mode"] == "live"
    assert mini.image.shape == with_layout.image.shape  # same scale as the top-down view
    assert not np.array_equal(mini.image, with_layout.image)
    other = session.step(np.full_like(frame, 255))  # the camera image is not in it
    assert np.array_equal(other.image, mini.image)
    assert np.array_equal(session.step(None).image, mini.image)  # camera down
    session.dispatch("view.terrain")  # "Layout" off: the minimap keeps the layout
    assert np.array_equal(session.step(frame).image, mini.image)
    session.dispatch("view.grid")
    assert not np.array_equal(session.step(frame).image, mini.image)
    session.dispatch("map.layout", name=None)  # no layout: off, and it stays off with the next layout
    assert session.step(frame).state["minimap"] is False
    session.dispatch("map.layout", name="Small")
    assert session.step(frame).state["minimap"] is False


def test_minimap_shows_named_bases_and_takes_clicks(tmp_path, named):
    set_wounds(tmp_path, {"Nob": 5})
    write_map(tmp_path / "maps")
    session, frame, state = named_session(tmp_path, named)
    session.dispatch("map.layout", name="Small")
    session.dispatch("view.minimap")
    mini = session.step(frame)
    assert mini.state["minimap"] is True and mini.state["mode"] == "snapshot"
    nob = base_of(mini.state, "Nob")
    x, y = (int(round(v)) for v in mini.state["bases"][nob]["position_px"])
    up = int(0.35 * mini.state["bases"][nob]["diameter_mm"] * mini.scale)  # the name covers the centre
    assert tuple(int(v) for v in mini.image[y - up, x]) == live_mod.TEAM_COLORS[0]  # a full disc in the team colour
    click_base(session, mini.state, nob)
    assert session.state()["selected"] == nob
    session.dispatch("wounds.minus")
    assert not np.array_equal(session.step(frame).image, mini.image)
    session.dispatch("snapshot.back")  # live video under the minimap: still the minimap
    live = session.step(frame)
    assert live.state["mode"] == "live" and live.state["minimap"] is True
    session.dispatch("view.minimap")
    off = session.step(frame)
    assert off.state["minimap"] is False and off.state["bases"][nob]["position_px"] is not None


def test_minimap_off_gives_the_snapshot_its_own_scale_again(tmp_path, named):
    write_map(tmp_path / "maps")
    session, frame, state = named_session(tmp_path, named)
    session.dispatch("view.top_down")  # the snapshot as the camera frame
    before = session.step(frame)
    session.dispatch("map.layout", name="Small")
    with_layout = session.step(frame)
    session.dispatch("view.minimap")
    session.step(frame)
    session.dispatch("view.minimap")
    after = session.step(frame)
    assert after.scale == before.scale and np.array_equal(after.image, with_layout.image)
    assert after.state["bases"] == with_layout.state["bases"]


# --- game state: wounds and openings ----------------------------------------------


def set_wounds(tmp_path, wounds: dict[str, int]) -> None:
    team = tmp_path / "gallery" / "orks"
    idt.save_team_file(team, "orks", idt.load_team_file(team), wounds)


def named_session(tmp_path, named):
    cfg, frame = named
    session = make_session(tmp_path, cfg, frame, teams=("orks",))
    settled(session, frame)
    session.dispatch("snapshot.name")
    return session, frame, session.step(frame).state


def base_of(state, name: str) -> int:
    return next(b["index"] for b in state["bases"] if b["label"] == f"orks/{name}")


def test_wounds_start_at_the_maximum_and_stay_in_range(tmp_path, named):
    set_wounds(tmp_path, {"Nob": 5, "Burna": 12})
    session, frame, state = named_session(tmp_path, named)
    nob = base_of(state, "Nob")
    assert state["bases"][nob]["wounds"] == 5 and state["bases"][nob]["max_wounds"] == 5
    assert state["selected_wounds"] is None and not ACTIONS_BY_ID["wounds.minus"].enabled(state)
    full = session.step(frame).image
    session.dispatch("base.select", index=nob)
    s = session.state()
    assert s["selected_wounds"] == {"label": "orks/Nob", "name": "Nob", "current": 5, "max": 5}
    assert ACTIONS_BY_ID["wounds.plus"].enabled(s) and ACTIONS_BY_ID["wounds.set"].enabled(s)
    session.dispatch("wounds.minus")
    session.dispatch("wounds.minus")
    assert session.state()["selected_wounds"]["current"] == 3
    session.dispatch("wounds.set", value=99)
    assert session.state()["selected_wounds"]["current"] == 5
    session.dispatch("wounds.set", value=-4)
    assert session.state()["selected_wounds"]["current"] == 0
    session.dispatch("wounds.plus")
    s = session.state()
    assert s["selected_wounds"]["current"] == 1 and s["bases"][nob]["wounds"] == 1
    session.dispatch("base.deselect")
    assert not np.array_equal(session.step(frame).image, full)  # the health bar is shorter
    rows = {r["name"]: r for r in s["rosters"][0]["operatives"]}
    assert (rows["Nob"]["wounds"], rows["Nob"]["max_wounds"]) == (1, 5)
    assert (rows["Burna"]["wounds"], rows["Burna"]["max_wounds"]) == (12, 12)
    assert (rows["Grot"]["wounds"], rows["Grot"]["max_wounds"]) == (None, None)
    session.dispatch("snapshot.name")  # a new snapshot keeps the wounds
    assert session.step(frame).state["bases"][nob]["wounds"] == 1
    assert_json_safe(session.state())


def test_wounds_of_an_operative_with_no_maximum_give_a_warning(tmp_path, named):
    set_wounds(tmp_path, {"Nob": 5})
    session, frame, state = named_session(tmp_path, named)
    session.dispatch("base.select", index=base_of(state, "Grot"))
    s = session.state()
    assert s["selected_wounds"] == {"label": "orks/Grot", "name": "Grot", "current": None, "max": None}
    assert not ACTIONS_BY_ID["wounds.minus"].enabled(s)
    session.dispatch("wounds.minus")
    s = session.state()
    assert s["notice"]["level"] == "error" and "no maximum wounds" in s["notice"]["text"]
    assert s["selected_wounds"]["current"] is None


def test_wounds_need_a_named_base(tmp_path):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    session, state = bases_snapshot(tmp_path, cfg, frame)
    click_base(session, state, 0)
    s = session.state()
    assert s["selected_wounds"] is None and s["bases"][0]["wounds"] is None
    session.dispatch("wounds.plus")
    assert "named operative" in session.state()["notice"]["text"]


def test_hatchway_opens_and_closes_and_a_new_layout_closes_it(tmp_path):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    write_map(tmp_path / "maps")
    session = make_session(tmp_path, cfg, frame)
    session.dispatch("map.layout", name="Small")
    closed = settled(session, frame)
    assert closed.state["openings"] == [{"index": 0, "kind": session.openings[0].kind, "open": False}]
    assert not ACTIONS_BY_ID["opening.toggle"].enabled(closed.state)
    session.dispatch("opening.toggle")  # nothing selected: a hint, no change
    assert session.state()["openings"][0]["open"] is False
    x, y = opening_px(session)
    session.dispatch("board.click", x=x, y=y)
    assert ACTIONS_BY_ID["opening.toggle"].enabled(session.state())
    session.dispatch("opening.toggle")
    s = session.state()
    assert s["selected_opening"]["open"] is True and s["openings"][0]["open"] is True
    session.dispatch("base.deselect")
    opened = session.step(frame)
    assert not np.array_equal(opened.image, closed.image)  # the floor shows through
    session.dispatch("board.click", x=x, y=y)
    session.dispatch("opening.toggle")
    session.dispatch("base.deselect")
    assert np.array_equal(session.step(frame).image, closed.image)
    session.dispatch("board.click", x=x, y=y)
    session.dispatch("opening.toggle")
    session.dispatch("map.layout", name="Small")  # the game starts again with all closed
    assert session.state()["openings"][0]["open"] is False


# A wall with a hatchway between Nob (300, 350) and Burna (450, 200): x = 381, y = 241.3 to 292.1 (1 in grid).
# The straight line goes through the hatchway.
WALL_MAP = """schema = 1
name = "Wall"
terrain = "Test World"
pillars = [[15, 9.5], [15, 11.5]]

[[wall]]
key = "A2"
from = [15, 9.5]
to = [15, 11.5]
opening = true
"""


def wall_session(tmp_path):
    """Bases snapshot on the layout `Wall`; indexes of Nob and Burna."""
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    write_map(tmp_path / "maps", file="wall.toml", text=WALL_MAP)
    session = make_session(tmp_path, cfg, frame)
    session.dispatch("map.layout", name="Wall")
    settled(session, frame)
    session.dispatch("snapshot.bases")
    state = session.step(frame).state
    index = {game_name(session, i): i for i in range(len(state["bases"]))}
    return session, frame, state, index["Nob"], index["Burna"]


def test_measure_around_walls_is_on_with_a_layout_and_off_without(tmp_path):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    write_map(tmp_path / "maps", file="wall.toml", text=WALL_MAP)
    session = make_session(tmp_path, cfg, frame)
    action = ACTIONS_BY_ID["measure.walls"]
    s = session.state()
    assert s["around_walls"] is False and not action.enabled(s) and action.key == "w"
    session.dispatch("map.layout", name="Wall")
    s = session.state()
    assert s["around_walls"] is True and action.enabled(s) and action.checked(s)
    session.dispatch("measure.walls")
    assert session.state()["around_walls"] is False
    session.dispatch("map.layout", name="Wall")  # each layout load turns it on
    assert session.state()["around_walls"] is True
    session.dispatch("map.layout", name=None)
    assert not action.enabled(session.state())


def test_distance_goes_around_a_wall_and_through_an_open_hatchway(tmp_path):
    session, frame, state, nob, burna = wall_session(tmp_path)
    straight = np.linalg.norm(np.subtract(GAME["Nob"][0], GAME["Burna"][0])) - 0.5 * SIZES["Nob"] - 0.5 * SIZES["Burna"]
    click_base(session, state, nob)
    click_base(session, state, burna, "board.select")
    around = session.step(frame)
    d = around.state["distance"]
    assert_json_safe(around.state)
    assert d["around"] is True and straight + 5.0 < d["mm"] < straight + 12.0  # around the pillar at the wall end
    assert d["in"] == pytest.approx(d["mm"] / 25.4, abs=1e-3)

    session.dispatch("measure.walls")
    plain = session.step(frame)
    assert plain.state["distance"]["around"] is False and abs(plain.state["distance"]["mm"] - straight) < 2.0
    assert not np.array_equal(plain.image, around.image)
    session.dispatch("measure.walls")
    assert np.array_equal(session.step(frame).image, around.image)

    x, y = opening_px(session)
    session.dispatch("board.click", x=x, y=y)
    session.dispatch("opening.toggle")
    click_base(session, state, nob)
    click_base(session, state, burna, "board.select")
    through = session.step(frame).state["distance"]
    assert through["around"] is True and abs(through["mm"] - straight) < 2.0  # through the gap

    session.dispatch("map.layout", name=None)  # no layout: no walls
    assert session.state()["distance"]["around"] is False


def test_ring_around_walls_is_a_filled_area_that_stops_at_the_wall(tmp_path):
    session, frame, state, nob, _ = wall_session(tmp_path)
    cfg = session.cfg
    click_base(session, state, nob)
    before = session.step(frame).image.copy()
    session.dispatch("ring.4")
    area = session.step(frame).image.copy()
    outline = session._wall_range[1]
    session.dispatch("measure.walls")
    ring = session.step(frame).image.copy()

    def changed(image, at_mm) -> bool:
        x, y = np.rint((np.asarray(at_mm) + cfg.rectify_margin_mm) * cfg.rectify_px_per_mm * session.win.scale).astype(int)
        return bool(np.any(image[y, x] != before[y, x]))

    center = np.asarray(GAME["Nob"][0])
    assert changed(area, center + (-60.0, 0.0)) and not changed(ring, center + (-60.0, 0.0))  # the fill, open floor
    behind = (386.0, 268.0)  # in the plain ring, but behind the wall
    assert np.linalg.norm(behind - center) < 0.5 * SIZES["Nob"] + 4 * 25.4
    assert not walls.contains(outline, behind) and walls.contains(outline, center + (-60.0, 0.0))
    assert np.count_nonzero(np.any(area != before, axis=2)) > 5 * np.count_nonzero(np.any(ring != before, axis=2))

    session.dispatch("measure.walls")
    session.dispatch("ring.clear")
    assert np.array_equal(session.step(frame).image, before)


def hover(session, at_mm) -> None:
    """Put the mouse pointer on a board position."""
    p = np.asarray(session._to_px(np.asarray(at_mm, float).reshape(1, 2)), float).reshape(2) * session.win.scale
    session.dispatch("board.hover", x=float(p[0]), y=float(p[1]))


def test_movement_area_and_increments_to_the_pointer(tmp_path):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    session, state = bases_snapshot(tmp_path, cfg, frame)
    action = ACTIONS_BY_ID["measure.move"]
    assert state["movement"] is False and state["move"] is None and action.key == "m" and action.enabled(state)
    nob = next(i for i in range(len(state["bases"])) if game_name(session, i) == "Nob")
    click_base(session, state, nob)
    session.dispatch("ring.3")
    ring = session.step(frame).image.copy()
    hover(session, np.add(GAME["Nob"][0], (60.0, 0.0)))
    assert session.state()["move"] is None and np.array_equal(session.step(frame).image, ring)  # off: no route

    session.dispatch("measure.move")
    live = session.step(frame)
    assert_json_safe(live.state)
    assert live.state["movement"] is True and action.checked(live.state)
    assert live.state["move"] == {"in": 3, "increments": [3]}  # 60 mm in a straight line
    area = live.image.copy()
    assert not np.array_equal(area, ring)

    hover(session, np.add(GAME["Nob"][0], (0.0, -20.0)))
    moved = session.step(frame)
    assert moved.state["move"] == {"in": 1, "increments": [1]} and not np.array_equal(moved.image, area)
    hover(session, np.add(GAME["Nob"][0], (0.0, 3 * 25.4 + 0.5 * SIZES["Nob"] + 30.0)))  # out of reach
    assert session.state()["move"] is None
    session.dispatch("board.hover", x=None, y=None)  # the pointer left the image
    assert session.state()["move"] is None
    session.dispatch("ring.clear")
    hover(session, np.add(GAME["Nob"][0], (60.0, 0.0)))
    assert session.state()["move"] is None  # no ring: no movement area


def test_movement_goes_around_a_wall_and_through_an_open_hatchway(tmp_path):
    session, frame, state, nob, _ = wall_session(tmp_path)
    goal = (421.5, 225.0)  # the straight line to it goes through the hatchway, but the pillars are too near for the base
    click_base(session, state, nob)
    session.dispatch("measure.move")
    session.dispatch("ring.9")
    hover(session, goal)
    around = session.step(frame).state["move"]
    assert len(around["increments"]) >= 2 and around["in"] == sum(around["increments"]) >= 7

    x, y = opening_px(session)
    session.dispatch("board.click", x=x, y=y)
    session.dispatch("opening.toggle")
    click_base(session, state, nob)
    session.dispatch("ring.9")
    hover(session, goal)
    field = session._field(session._found())[0]
    assert field.barrier.any()  # the open hatchway is in the movement field
    through = session.step(frame).state["move"]
    assert through["in"] == sum(through["increments"]) and through["in"] <= around["in"]  # this wall is short: the way around is as good

    session.dispatch("map.layout", name=None)  # no terrain: straight
    assert session.state()["move"] == {"in": 7, "increments": [7]}


def test_only_named_bases_of_the_other_team_block_the_movement(tmp_path):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    session, state = bases_snapshot(tmp_path, cfg, frame)
    found = session._found()
    session.selected = 0
    assert session._enemies(found) == []  # no names
    for labels, blocking in (
        (["red/a", "blue/b", "red/c"], [1]),
        (["red/a", None, "blue/c"], [2]),
        ([None, "blue/b", "red/c"], []),  # the selected base has no name
        (["red/a", "red/b", "red/c"], []),
    ):
        session.session = SimpleNamespace(idents=[SimpleNamespace(label=label) for label in labels])
        got = session._enemies(found)
        assert [center for center, _ in got] == [tuple(float(v) for v in found[i].center_mm) for i in blocking]
        assert [size for _, size in got] == [found[i].diameter_mm for i in blocking]
    session.session = None


def test_no_stop_on_another_base_in_the_movement_area(tmp_path):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    session, state = bases_snapshot(tmp_path, cfg, frame)
    index = {game_name(session, i): i for i in range(len(state["bases"]))}
    click_base(session, state, index["Nob"])
    session.dispatch("measure.move")
    session.dispatch("ring.9")
    session.step(frame)
    field, outline = session._field(session._found())
    burna = np.asarray(GAME["Burna"][0])  # 212 mm away: in reach, no name, so it does not block
    assert not walls.contains(outline, burna) and field.cost(burna) is None
    beside = burna + (0.0, 0.5 * SIZES["Burna"] + 0.5 * SIZES["Nob"] + 3.0)
    assert walls.contains(outline, beside) and field.cost(beside) is not None
    hover(session, burna)
    assert session.state()["move"] is None


# --- control marks ------------------------------------------------------------------

# Nob and Grot rims 20 mm apart; Burna rim 20 mm from the edge of the objective marker.
CLOSE = {"Nob": ((300.0, 350.0), 120.0), "Grot": ((352.5, 350.0), 200.0), "Burna": ((450.0, 200.0), 300.0)}
OBJECTIVE_MAP = """schema = 1
name = "Objective"
terrain = "Test World"

[[objective]]
at_mm = [450, 256]
"""


def close_session(tmp_path, other: str | None = "Grot"):
    """Named snapshot of CLOSE; `other` goes to a second team."""
    cfg = ident_cfg()
    gallery = make_gallery(tmp_path, cfg)
    enroll_from(gallery, cfg, paint_board(cfg, TRAIN), {n: c for n, (c, _) in TRAIN.items()})
    frame = paint_board(cfg, CLOSE)
    session = make_session(tmp_path, cfg, frame, teams=("orks",))
    settled(session, frame)
    session.dispatch("snapshot.name")
    found = session._found()
    index = {min(CLOSE, key=lambda n: np.linalg.norm(np.subtract(CLOSE[n][0], c.center_mm))): i for i, c in enumerate(found)}
    assert len(index) == 3
    for name, i in index.items():
        team = "elves" if name == other else "orks"
        session.session.idents[i] = replace(session.session.idents[i], label=f"{team}/{name}")
    session.dirty = True
    return session, frame, index


def yellow_pixels(image) -> int:
    return int(np.sum(np.all(image == overlay.MARK_COLOR, axis=2)))


def test_control_marks_need_named_operatives(tmp_path):
    cfg = ident_cfg()
    frame = paint_board(cfg, CLOSE)
    session = make_session(tmp_path, cfg, frame)
    assert not ACTIONS_BY_ID["view.marks"].enabled(settled(session, frame).state)
    session.dispatch("snapshot.bases")  # bases without names
    s = session.step(frame).state
    assert s["bases"] and not ACTIONS_BY_ID["view.marks"].enabled(s) and s["control"] is None


def test_control_marks_are_on_at_the_start_and_show_melee(tmp_path):
    session, frame, index = close_session(tmp_path)
    assert ACTIONS_BY_ID["view.marks"].enabled(session.state())
    live = session.step(frame)
    assert live.state["marks"] is True and ACTIONS_BY_ID["view.marks"].checked(live.state)
    assert live.state["control"] == {"melee": [sorted([index["Nob"], index["Grot"]])], "objectives": []}
    assert_json_safe(live.state)
    assert yellow_pixels(live.image) > 50
    session.dispatch("view.marks")
    off = session.step(frame)
    assert off.state["marks"] is False and off.state["control"] is None and yellow_pixels(off.image) == 0


def test_no_melee_in_one_team(tmp_path):
    session, frame, _ = close_session(tmp_path, other=None)
    live = session.step(frame)
    assert live.state["control"] == {"melee": [], "objectives": []} and yellow_pixels(live.image) == 0


def test_objective_arrows_need_a_layout(tmp_path):
    write_map(tmp_path / "maps", file="objective.toml", text=OBJECTIVE_MAP)
    session, frame, index = close_session(tmp_path)
    without = session.step(frame)
    assert without.state["control"]["objectives"] == []
    session.dispatch("map.layout", name="Objective")
    session.dispatch("view.terrain")  # the arrows only
    session.dispatch("view.grid")
    with_layout = session.step(frame)
    assert with_layout.state["control"]["objectives"] == [[index["Burna"], 0]]
    assert yellow_pixels(with_layout.image) > yellow_pixels(without.image)


def test_arrow_heads_go_to_the_objective_or_around_the_base():
    far = overlay.objective_arrows_mm((0.0, 0.0), 32.0, (0.0, 56.0))  # 40 mm from rim to centre
    assert len(far) == 6 and all(h[1][0] == pytest.approx(0.0) for h in far)
    tips = [h[1][1] for h in far]
    assert tips[0] == pytest.approx(16.0 + overlay.ARROW_LEN_MM) and max(tips) <= 56.0
    assert all(h[1][1] > h[0][1] for h in far)  # pointing to the objective
    near = overlay.objective_arrows_mm((0.0, 0.0), 32.0, (5.0, 5.0))  # objective under the base
    assert len(near) == 4
    for h in near:
        tip, back = h[1], 0.5 * (h[0] + h[2])
        assert np.linalg.norm(tip) == pytest.approx(17.0) and np.linalg.norm(back) > np.linalg.norm(tip)  # to the rim
        assert abs(tip[0]) == pytest.approx(abs(tip[1]))  # X fashion


# --- score counter -----------------------------------------------------------------


def test_score_counter_of_each_side(tmp_path):
    from battlesight import ident as idt
    from battlesight import score as sc

    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    for team, archetypes in (("orks", ["Seek and Destroy", "Security"]), ("elves", ["Recon"])):
        (tmp_path / "gallery" / team).mkdir(parents=True)
        idt.save_team_file(tmp_path / "gallery" / team, team, {}, None, idt.TeamInfo(tuple(archetypes)))
    session = make_session(tmp_path, cfg, frame, teams=("orks",))
    s = session.state()
    assert_json_safe(s)
    left, right = s["scores"]
    assert left["tac_ops"] == sc.tac_ops_for(["Seek and Destroy", "Security"]) and right["tac_ops"] == []
    assert left["cp"] == right["cp"] == 3 and left["total"] == 0 and left["primary"] == "hidden"

    session.dispatch("score.skull", slot=0, category="crit", tp=2, skull=2)
    session.dispatch("score.skull", slot=0, category="kill", tp=None, skull=3)
    session.dispatch("score.primary", slot=0, value="crit")
    session.dispatch("score.cp", slot=0, delta=-1)
    session.dispatch("score.tac_op", slot=0, name="Rout")
    left = session.state()["scores"][0]
    assert left["cells"]["crit"] == [2, 0, 0] and left["points"] == {"crit": 2, "tac": 0, "kill": 3}
    assert left["primary_bonus"] == 1 and left["total"] == 6 and left["cp"] == 2 and left["tac_op"] == "Rout"
    assert session.state()["scores"][1]["total"] == 0

    session.dispatch("score.tac_op", slot=0, name="Flank")  # a Recon op: not for orks
    assert "not a Tac Op" in session.state()["notice"]["text"] and session.state()["scores"][0]["tac_op"] == "Rout"
    session.dispatch("score.skull", slot=0, category="crit", tp=5, skull=1)
    assert "turning point" in session.state()["notice"]["text"]
    session.dispatch("score.skull", slot=2, category="crit", tp=2, skull=1)
    assert "no team slot" in session.state()["notice"]["text"]

    session.dispatch("team.set", slot=0, name="elves")  # the score stays, the Tac Op of orks goes
    left = session.state()["scores"][0]
    assert left["total"] == 6 and left["tac_op"] is None and left["tac_ops"] == list(sc.TAC_OPS["Recon"])


def test_select_an_operative_from_a_team_list(tmp_path, named):
    from battlesight import ident as idt

    cfg, frame = named
    team = tmp_path / "gallery" / "orks"
    idt.save_team_file(team, "orks", idt.load_team_file(team), {"Nob": 5})
    session = make_session(tmp_path, cfg, frame, teams=("orks",))
    settled(session, frame)
    session.dispatch("operative.select", label="orks/Nob")  # no snapshot: the operative alone
    s = session.state()
    assert s["selected"] is None and s["selected_operative"] == "orks/Nob" and s["selected_label"] == "orks/Nob"
    assert s["selected_wounds"] == {"label": "orks/Nob", "name": "Nob", "current": 5, "max": 5}
    assert next(a for a in ACTIONS if a.id == "base.deselect").enabled(s)
    session.dispatch("wounds.minus")
    assert session.state()["selected_wounds"]["current"] == 4
    session.dispatch("base.deselect")
    assert session.state()["selected_operative"] is None and session.state()["selected_wounds"] is None
    session.dispatch("operative.select", label="orks/Nobody")
    assert "not in a team list" in session.state()["notice"]["text"] and session.state()["selected_operative"] is None

    session.dispatch("snapshot.name")
    s = session.step(frame).state
    nob = next(b["index"] for b in s["bases"] if b["label"] == "orks/Nob")
    session.dispatch("operative.select", label="orks/Nob")  # identified: its base
    s = session.state()
    assert s["selected"] == nob and s["selected_operative"] is None and s["selected_label"] == "orks/Nob"


def test_team_list_remove_and_add_operatives(tmp_path, named):
    cfg, frame = named
    session = make_session(tmp_path, cfg, frame, teams=("orks",))
    settled(session, frame)
    names = lambda: [r["name"] for r in session.state()["rosters"][0]["operatives"]]  # noqa: E731
    assert names() == ["Nob", "Grot", "Burna"] and session.state()["rosters"][0]["missing"] == []

    session.dispatch("roster.remove", slot=0)  # nothing selected
    assert "select an operative" in session.state()["notice"]["text"] and len(names()) == 3
    session.dispatch("operative.select", label="orks/Nob")
    session.dispatch("roster.remove", slot=1)  # Nob is not on the right side
    assert len(names()) == 3
    session.dispatch("roster.remove", slot=0)
    s = session.state()
    assert names() == ["Grot", "Burna"] and s["rosters"][0]["missing"] == [{"label": "orks/Nob", "name": "Nob", "max": 1, "copies_in_play": 0}]
    assert s["selected_label"] is None

    session.dispatch("snapshot.name")  # Nob is out of play: no base gets its name
    s = session.step(frame).state
    assert session.gallery.out_of_play == {"orks/Nob"}
    assert "orks/Nob" not in {b["label"] for b in s["bases"]}
    assert {b["label"] for b in s["bases"]} >= {"orks/Grot", "orks/Burna"}

    session.dispatch("roster.add", slot=0, label="orks/Grot")  # already in the list
    assert "cannot be added" in session.state()["notice"]["text"]
    session.dispatch("roster.add", slot=0, label="orks/Nob")  # back at the end
    assert names() == ["Grot", "Burna", "Nob"] and session.gallery.out_of_play == set()
    assert session.state()["rosters"][0]["missing"] == []

    session.dispatch("operative.select", label="orks/Grot")
    session.dispatch("roster.remove", slot=0)
    session.dispatch("team.set", slot=0, name=None)
    session.dispatch("team.set", slot=0, name="orks")  # a team change gives all operatives again
    assert names() == ["Nob", "Grot", "Burna"]


# --- copies of one operative -------------------------------------------------------


def paint_copies(cfg, placed: list[tuple[str, tuple[float, float], float]]):
    """Board with painted operatives; a name can come more than once (copies)."""
    from .conftest import Base
    from .test_ident import PAINT, SIZES

    img, mm2px = render_board(cfg, px_per_mm=2.0, bases=tuple(Base(c, SIZES[n]) for n, c, _ in placed))
    for name, c, turn in placed:
        body, detail = PAINT[name]
        r = 0.5 * SIZES[name]
        p = mm2px(c)
        cv2.circle(img, (int(p[0]), int(p[1])), int((r - 3) * 2.0), body, -1, cv2.LINE_AA)
        a = np.deg2rad(turn)
        q = mm2px(np.array(c) + 0.4 * r * np.array([np.cos(a), np.sin(a)]))
        cv2.circle(img, (int(q[0]), int(q[1])), int(0.3 * r * 2.0), detail, -1, cv2.LINE_AA)
    return img


A, B, C = (150.0, 450.0), (500.0, 150.0), (520.0, 300.0)
OTHERS = [("Nob", (480.0, 620.0), 30.0), ("Burna", (120.0, 150.0), 60.0)]


@pytest.fixture
def grots(tmp_path, named):
    """The orks gallery with `max = 3` for Grot, and a session with two Grots in the team list."""
    cfg, _ = named
    team = tmp_path / "gallery" / "orks"
    idt.save_team_file(team, "orks", idt.load_team_file(team), {"Grot": 5}, None, None, {"Grot": 3})
    frame = paint_copies(cfg, [("Grot", A, 0.0), ("Grot", B, 90.0), *OTHERS])
    session = make_session(tmp_path, cfg, frame, teams=("orks",))
    settled(session, frame)
    session.dispatch("roster.add", slot=0, label="orks/Grot")
    return cfg, session


def snap(session, frame) -> dict:
    settled(session, frame)
    session.dispatch("snapshot.name")
    return session.step(frame).state


def grot_bases(session, state) -> dict[tuple[float, float], dict]:
    """Grot bases of the snapshot by their centre (mm, rounded to 10 mm)."""
    found = session._found()
    return {tuple(np.round(np.asarray(found[b["index"]].center_mm) / 10.0) * 10.0): b for b in state["bases"] if b["type_label"] == "orks/Grot"}


def test_team_list_with_copies(grots):
    _, session = grots
    rows = session.state()["rosters"][0]
    assert [r["name"] for r in rows["operatives"]] == ["Nob", "Grot 1", "Burna", "Grot 2"]
    assert [r["label"] for r in rows["operatives"]][-1] == "orks/Grot#2"
    assert rows["missing"] == [{"label": "orks/Grot", "name": "Grot", "max": 3, "copies_in_play": 2}]
    session.dispatch("roster.add", slot=0, label="orks/Grot")
    s = session.state()
    assert s["rosters"][0]["operatives"][-1]["name"] == "Grot 3" and s["rosters"][0]["missing"] == []
    session.dispatch("roster.add", slot=0, label="orks/Grot")
    assert "all 3 copies" in session.state()["notice"]["text"]
    session.dispatch("roster.add", slot=0, label="orks/Nob#2")
    assert "cannot be added" in session.state()["notice"]["text"]
    # Remove copy 2: its wounds go; it comes back with all wounds and the same number.
    session.dispatch("wounds.set", value=1, label="orks/Grot#2")
    session.dispatch("operative.select", label="orks/Grot#2")
    session.dispatch("roster.remove", slot=0)
    names = [r["name"] for r in session.state()["rosters"][0]["operatives"]]
    assert names == ["Nob", "Grot 1", "Burna", "Grot 3"] and "orks/Grot#2" not in session.wounds
    session.dispatch("roster.add", slot=0, label="orks/Grot")
    rows = session.state()["rosters"][0]["operatives"]
    assert rows[-1]["name"] == "Grot 2" and rows[-1]["wounds"] == 5
    assert session.gallery is None or session.gallery.copies["orks/Grot"] == 3


def test_copies_get_numbers_by_position_and_keep_them(grots):
    cfg, session = grots
    state = snap(session, paint_copies(cfg, [("Grot", A, 0.0), ("Grot", B, 90.0), *OTHERS]))
    assert_json_safe(state)
    grot = grot_bases(session, state)
    assert grot[A]["label"] == "orks/Grot#1" and grot[B]["label"] == "orks/Grot#2"  # first snapshot: nearest the top-left corner first
    assert grot[B]["name"] == "Grot 2" and grot[B]["copy"] == 2 and grot[B]["sure"]
    session.dispatch("wounds.set", value=2, label="orks/Grot#2")
    # Grot 2 moves from B to C, Grot 1 stays: no question.
    state = snap(session, paint_copies(cfg, [("Grot", A, 10.0), ("Grot", C, 180.0), *OTHERS]))
    grot = grot_bases(session, state)
    assert grot[A]["label"] == "orks/Grot#1" and grot[C]["label"] == "orks/Grot#2"
    assert grot[A]["sure"] and grot[C]["sure"] and grot[C]["wounds"] == 2
    assert np.allclose(session.positions["orks/Grot#2"], C, atol=2.0)


def equidistant(a, c, off: float = 50.0) -> tuple[tuple[float, float], tuple[float, float]]:
    """Two points at the same distance from a and from c: a swap of the copies there costs nothing."""
    mid, d = 0.5 * (np.asarray(a, float) + c), np.asarray(c, float) - a
    n = np.array([-d[1], d[0]]) / np.hypot(*d)
    return tuple(np.round(mid + off * n)), tuple(np.round(mid - off * n))


def test_a_possible_swap_is_unsure_only_when_the_states_differ(grots):
    cfg, session = grots
    snap(session, paint_copies(cfg, [("Grot", A, 0.0), ("Grot", C, 90.0), *OTHERS]))
    p, q = equidistant(A, C)
    state = snap(session, paint_copies(cfg, [("Grot", p, 0.0), ("Grot", q, 90.0), *OTHERS]))
    assert all(b["sure"] for b in grot_bases(session, state).values())  # same wounds: a swap changes nothing
    before = {k: v.copy() for k, v in session.positions.items()}
    session.dispatch("wounds.set", value=1, label="orks/Grot#1")
    p2, q2 = equidistant(p, q)
    state = snap(session, paint_copies(cfg, [("Grot", p2, 0.0), ("Grot", q2, 90.0), *OTHERS]))
    grot = list(grot_bases(session, state).values())
    assert len(grot) == 2 and not any(b["sure"] for b in grot)
    assert all(np.allclose(session.positions[k], v) for k, v in before.items())  # unsure: the positions stay
    # One click fixes it: choose the other copy on one base, the bases swap.
    first, second = grot
    other = "orks/Grot#2" if first["label"] == "orks/Grot#1" else "orks/Grot#1"
    session.dispatch("base.choose", label=other, index=first["index"])
    after = {b["index"]: b for b in session.state()["bases"]}
    assert after[first["index"]]["label"] == other and after[second["index"]]["label"] == first["label"]
    assert after[first["index"]]["confirmed"] and after[second["index"]]["confirmed"]
    found = session._found()
    assert np.allclose(session.positions[other], found[first["index"]].center_mm)
    assert np.allclose(session.positions[first["label"]], found[second["index"]].center_mm)


def test_name_menu_lists_each_copy_with_its_wounds(grots):
    cfg, session = grots
    state = snap(session, paint_copies(cfg, [("Grot", A, 0.0), ("Grot", B, 90.0), *OTHERS]))
    session.dispatch("wounds.set", value=3, label="orks/Grot#2")
    index = grot_bases(session, state)[A]["index"]
    session.dispatch("base.select", index=index)
    menu = session.state()["candidates"]
    copies = [c for c in menu if c["label"].startswith("orks/Grot")]
    assert [(c["name"], c["copy"], c["wounds"]) for c in copies] == [("Grot 1", 1, 5), ("Grot 2", 2, 3)]
    from battlesight.ui.parts import candidate_text

    assert candidate_text(copies[1]).startswith("orks/Grot 2 (3/5)")
    # A type label picks the copy that the base has.
    session.dispatch("base.choose", label="orks/Grot", index=index)
    assert session.state()["bases"][index]["label"] == "orks/Grot#1"


def test_movement_shows_the_melee_zone_and_avoid_melee_keeps_out_of_it(tmp_path):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    session, state = bases_snapshot(tmp_path, cfg, frame)
    index = {game_name(session, i): i for i in range(len(state["bases"]))}
    nob, burna = index["Nob"], index["Burna"]
    avoid = ACTIONS_BY_ID["measure.avoid_melee"]
    assert state["avoid_melee"] is False and not avoid.enabled(state) and state["move_melee"] == []
    click_base(session, state, nob)
    session.dispatch("ring.9")
    plain = session.step(frame).image.copy()
    session._enemy_indexes = lambda found: [burna]  # as with names of two teams
    session.dispatch("measure.move")
    live = session.step(frame)
    assert avoid.enabled(live.state) and not avoid.checked(live.state)

    edge = np.asarray(GAME["Burna"][0]) + (0.5 * SIZES["Burna"] + 25.4, 0.0)  # on the dashed circle
    x, y = np.rint((edge + cfg.rectify_margin_mm) * cfg.rectify_px_per_mm * session.win.scale).astype(int)
    patch = live.image[y - 6 : y + 7, x - 6 : x + 7].reshape(-1, 3).astype(int)
    assert ((patch[:, 2] > 200) & (patch[:, 1] > 150) & (patch[:, 0] < 120)).any()  # yellow

    near = np.asarray(GAME["Burna"][0]) + (-50.0, 0.0)  # the rims are 2 mm apart there
    hover(session, near)
    melee = session.step(frame)
    assert_json_safe(melee.state)
    assert melee.state["move_melee"] == [burna] and melee.state["move"] is not None
    bx, by = np.rint((np.asarray(GAME["Burna"][0]) + cfg.rectify_margin_mm) * cfg.rectify_px_per_mm * session.win.scale).astype(int)
    sign = melee.image[by - 12 : by + 13, bx - 12 : bx + 13].reshape(-1, 3)
    assert (sign.min(axis=1) > 240).any() and not (live.image[by - 12 : by + 13, bx - 12 : bx + 13].reshape(-1, 3).min(axis=1) > 240).any()  # white swords
    hover(session, np.add(GAME["Nob"][0], (-40.0, 0.0)))  # away from the enemy
    assert session.state()["move_melee"] == []

    session.dispatch("measure.avoid_melee")
    hover(session, near)
    kept = session.step(frame)
    assert kept.state["avoid_melee"] is True and avoid.checked(kept.state)
    assert kept.state["move_melee"] == [] and not np.array_equal(kept.image, melee.image)
    field, outline = session._field(session._found())
    assert field.cost(near) is None and not walls.contains(outline, np.asarray(GAME["Burna"][0]) + (-30.0, 0.0))
    session.dispatch("measure.move")
    assert not avoid.enabled(session.state())
