"""Game core of the live app (`live.LiveSession`) on synthetic boards. No camera,
no window, no model download."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import battlesight.app as app
import battlesight.live as live_mod
from battlesight import ident as idt
from battlesight import colors, measure, overlay, walls
from battlesight import terrain as ter
from battlesight.calib import TagDetector
from battlesight.camera import CameraSettings, ImageSource
from battlesight.live import ACTIONS, PREVIEW_PAD_MM, LiveSession

from .conftest import render_board
from .test_ident import SIZES, enroll_from, ident_cfg, make_gallery, paint_board

TRAIN = {"Nob": ((150.0, 150.0), 0.0), "Grot": ((300.0, 200.0), 40.0), "Burna": ((450.0, 500.0), 90.0)}
GAME = {"Nob": ((300.0, 350.0), 120.0), "Grot": ((150.0, 500.0), 200.0), "Burna": ((450.0, 200.0), 300.0)}
SETTLE = 7  # more than settle_frames of the test config


def near_color(pixels: np.ndarray, color, tol: int = 40) -> np.ndarray:
    """Pixels (last axis BGR) within `tol` of `color` in each channel: an
    anti-aliased line of a colour of `data/color-settings.toml`, whatever the
    file says."""
    return np.abs(pixels.astype(int) - np.asarray(color, int)).max(axis=-1) <= tol


def make_session(tmp_path, cfg, frame, **over) -> LiveSession:
    size = (frame.shape[1], frame.shape[0])
    args = dict(gallery_root=tmp_path / "gallery", reference_path=tmp_path / "ref" / "empty.png", snapshot_dir=tmp_path / "snaps", maps_root=tmp_path / "maps", pieces_root=tmp_path / "terrain", saves_dir=tmp_path / "saves")
    args.update(over)
    return LiveSession(cfg, CameraSettings(), size, **args)


def settled(session: LiveSession, frame):
    for _ in range(SETTLE):
        live = session.step(frame)
    return live


@pytest.fixture(scope="session")
def named_template(tmp_path_factory):
    """Gallery with three operatives, enrolled once per test process."""
    root = tmp_path_factory.mktemp("named")
    cfg = ident_cfg()
    gallery = make_gallery(root, cfg)
    enroll_from(gallery, cfg, paint_board(cfg, TRAIN), {n: c for n, (c, _) in TRAIN.items()})
    return cfg, root / "gallery", paint_board(cfg, GAME)


@pytest.fixture
def named(tmp_path, named_template):
    """Gallery with three operatives in `tmp_path / "gallery"` (a copy of the
    session gallery, so a test can change it), and a game frame with them
    moved and turned."""
    cfg, gallery_dir, frame = named_template
    shutil.copytree(gallery_dir, tmp_path / "gallery")
    return cfg, frame.copy()


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
    assert s["selected"] == 1 and s["can_undo"]  # the named base stays selected
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
    assert s["bases"][2]["label"] == label and s["bases"][2]["confirmed"] and s["new_base"] is None and s["selected"] == 2
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
    s = session.state()
    assert s["selected"] == 0 and s["new_base"] is None and s["bases"][0]["confirmed"]  # the located operative stays selected
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


def test_ring_colors_the_other_bases_in_its_distance(tmp_path):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    session, state = bases_snapshot(tmp_path, cfg, frame)
    click_base(session, state, 0)
    plain = session.step(frame)
    assert plain.state["in_ring"] == []
    found = session._found()
    a = found[0]
    gaps = [measure.rim_gap(a.center_mm, a.diameter_mm, c.center_mm, c.diameter_mm)[0] for c in found]
    inches = next(n for n in range(1, 10) if any(g <= n * measure.MM_PER_INCH for g in gaps[1:]))
    session.dispatch(f"ring.{inches}")
    live = session.step(frame)
    expected = [i for i in range(1, len(found)) if gaps[i] <= inches * measure.MM_PER_INCH]
    assert live.state["in_ring"] == expected and expected
    assert_json_safe(live.state)
    session.dispatch("measure.move")
    assert session.step(frame).state["in_ring"] == []


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
    assert s["selected"] == 0 and s["target"] is None and s["bases"][0]["confirmed"]


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


def test_selected_opening_shows_its_ranges_and_the_bases_near_it(tmp_path):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    write_map(tmp_path / "maps")
    session = make_session(tmp_path, cfg, frame)
    session.dispatch("map.layout", name="Small")
    settled(session, frame)
    session.dispatch("snapshot.bases")
    plain = session.step(frame)
    assert plain.state["near_opening"] == []
    found, points = session._found(), session.openings[0].points
    # move one base to 1.5 in from the opening edge, so that one base is near it
    edge = points[np.argmin(points[:, 1])]
    found[0] = replace(found[0], center_mm=np.array([points[:, 0].mean(), edge[1] - 1.5 * measure.MM_PER_INCH - 0.5 * found[0].diameter_mm]))
    session.dirty = True
    x, y = opening_px(session)
    session.dispatch("board.click", x=x, y=y)
    shown = session.step(frame)
    near = shown.state["near_opening"]
    expected = [i for i, c in enumerate(found) if measure.polygon_distance(points, c.center_mm) - 0.5 * c.diameter_mm <= 2 * measure.MM_PER_INCH]
    assert 0 in near and near == expected
    assert_json_safe(shown.state)
    yellow = lambda img: int(np.sum(np.all(img == colors.RANGE_ZONE_COLOR, axis=2)))
    assert yellow(shown.image) > yellow(plain.image)
    session.dispatch("base.deselect")
    assert session.step(frame).state["near_opening"] == []


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
    from battlesight.colors import NO_TEAM_COLOR, TEAM_1_COLOR, TEAM_2_COLOR
    from battlesight.overlay import draw_idents

    class C:
        def __init__(self, x):
            self.center_mm = np.array([x, 50.0])
            self.diameter_mm = 32.0

    bases = [C(40.0), C(100.0), C(160.0)]
    idents = [idt.Ident("orks/Nob", 0.9, 0.5, sure=True), idt.Ident("elves/Ari", 0.9, 0.5, sure=True), idt.Ident(None, 0.0, 0.0)]
    img = np.zeros((120, 220, 3), np.uint8)
    draw_idents(img, bases, idents, lambda p: np.asarray(p, float), team_colors={"orks": TEAM_1_COLOR, "elves": TEAM_2_COLOR})
    for x, color in ((40, TEAM_1_COLOR), (100, TEAM_2_COLOR), (160, NO_TEAM_COLOR)):
        rim = img[30:51, x - 20 : x + 21].reshape(-1, 3).sum(axis=0).astype(float)  # upper half, no name there
        assert np.allclose(rim / rim.max(), np.array(color) / max(color), atol=0.05)  # anti-aliased: same hue
        name = img[60:120, x - 30 : x + 31].reshape(-1, 3)
        others = [c for c in (TEAM_1_COLOR, TEAM_2_COLOR, NO_TEAM_COLOR) if c != color]
        assert near_color(name, color).sum() > 5 and not any(near_color(name, c, tol=10).any() for c in others)  # the text in the team colour


def test_names_inside_put_the_name_in_the_middle_of_the_base():
    from battlesight.colors import TEAM_1_COLOR
    from battlesight.overlay import draw_idents

    class C:
        center_mm = np.array([100.0, 100.0])
        diameter_mm = 32.0

    x = idt.Ident("orks/Nob", 0.9, 0.5, sure=True)
    img = np.zeros((200, 200, 3), np.uint8)
    draw_idents(img, [C()], [x], lambda p: np.asarray(p, float), scores=False, team_colors={"orks": TEAM_1_COLOR}, inside=True)
    text = np.argwhere(near_color(img, TEAM_1_COLOR, tol=80))  # the text in the team colour
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
    session.dispatch("view.live")  # the live video in place of the minimap
    off = session.step(frame)
    assert off.state["mode"] == "live" and off.state["minimap"] is False and off.state["bases"][nob]["position_px"] is not None
    assert ACTIONS_BY_ID["view.live"].checked(off.state) and not ACTIONS_BY_ID["view.minimap"].checked(off.state)
    session.dispatch("view.minimap")
    session.dispatch("snapshot.back")  # Esc: from the minimap to the live video
    assert session.step(frame).state["minimap"] is False


def test_view_radio_buttons_show_live_snapshot_or_minimap(tmp_path, named):
    write_map(tmp_path / "maps")
    session, frame, state = named_session(tmp_path, named)
    radios = [ACTIONS_BY_ID[f"view.{m}"] for m in ("live", "snapshot", "minimap")]
    assert all(a.radio and a.control for a in radios) and not ACTIONS_BY_ID["snapshot.back"].control

    def shown():
        s = session.step(frame).state
        assert sum(a.checked(s) for a in radios) == 1
        return live_mod.shown_view(s)

    assert shown() == "snapshot"
    session.dispatch("view.live")
    assert shown() == "live"
    session.dispatch("view.snapshot")  # the last snapshot again, with its names
    assert shown() == "snapshot" and base_of(session.state(), "Nob") is not None
    session.dispatch("map.layout", name="Small")
    session.dispatch("view.minimap")
    assert shown() == "minimap"
    session.dispatch("view.snapshot")
    assert shown() == "snapshot"


def test_snapshot_radio_with_no_snapshot_pauses_the_video(tmp_path):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    session = make_session(tmp_path, cfg, frame)
    assert not ACTIONS_BY_ID["view.snapshot"].enabled(session.state())  # no camera frame yet
    session.dispatch("view.snapshot")
    assert session.state()["mode"] == "live"
    live = settled(session, frame)
    assert ACTIONS_BY_ID["view.snapshot"].enabled(live.state)
    session.dispatch("view.snapshot")
    paused = session.step(frame)
    assert paused.state["mode"] == "snapshot" and paused.state["bases"] == [] and paused.state["snapshot_ok"] is True
    assert np.array_equal(session.step(np.full_like(frame, 255)).image, paused.image)  # the video stops on that frame
    session.dispatch("view.live")
    session.dispatch("view.snapshot")  # the same paused frame again
    assert np.array_equal(session.step(np.full_like(frame, 255)).image, paused.image)


def test_minimap_off_gives_the_snapshot_its_own_scale_again(tmp_path, named):
    write_map(tmp_path / "maps")
    session, frame, state = named_session(tmp_path, named)
    session.dispatch("view.top_down")  # the snapshot as the camera frame
    before = session.step(frame)
    session.dispatch("map.layout", name="Small")
    with_layout = session.step(frame)
    session.dispatch("view.minimap")
    session.step(frame)
    session.dispatch("view.snapshot")
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


def test_dead_operatives_give_the_kill_op_of_the_other_side(tmp_path, named):
    set_wounds(tmp_path, {"Nob": 5, "Burna": 12, "Grot": 3})
    session, frame, state = named_session(tmp_path, named)
    assert state["team_slots"][0] == "orks" and state["team_slots"][1] is None
    assert [(x["kill_grade"], x["kill_bonus"]) for x in state["scores"]] == [(0, 0), (0, 0)]
    assert not any(r["dead"] for r in state["rosters"][0]["operatives"])
    session.dispatch("base.select", index=base_of(state, "Nob"))
    session.dispatch("wounds.set", value=0)
    s = session.state()
    rows = {r["name"]: r for r in s["rosters"][0]["operatives"]}
    assert rows["Nob"]["dead"] and not rows["Burna"]["dead"] and s["bases"][base_of(s, "Nob")]["dead"]
    # 3 orks: row 5 of the table, 1 dead = grade 1, and 1 more for the higher grade
    assert s["scores"][1]["points"]["kill"] == 2 and s["scores"][1]["total"] == 2 and s["scores"][0]["points"]["kill"] == 0
    team = tmp_path / "gallery" / "orks"
    idt.save_team_file(team, "orks", idt.load_team_file(team), {"Nob": 5, "Burna": 12, "Grot": 3}, None, ["Nob"])  # an expendable operative does not count
    session._team_ops.clear()
    assert session.state()["scores"][1]["points"]["kill"] == 0
    session.dispatch("wounds.set", value=1)
    assert session.state()["scores"][1]["points"]["kill"] == 0


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

    session.dispatch("ring.clear")  # a click with a ring would lock the route
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


LIGHT_MAP = """schema = 1
name = "Light"
terrain = "Test World"

[[light]]
key = "C5"
at_mm = [300, 300]
"""


def test_shoot_at_light_terrain_gives_cover_not_obscured(tmp_path):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    write_map(tmp_path / "maps", file="light.toml", text=LIGHT_MAP)
    pentagon = np.array([[-20, -20], [20, -20], [20, 20], [0, 25], [-20, 20]], float)
    folder = tmp_path / "terrain" / "test-world"
    folder.mkdir(parents=True)
    ter.write_piece(folder / "C5.toml", ter.Piece("C5", ter.piece_frame(pentagon)))
    session = make_session(tmp_path, cfg, frame)
    session.dispatch("map.layout", name="Light")
    assert [len(p) for p in session._light_outlines()] == [4]  # not identified: the card symbol
    session.light_poses[0] = ((300.0, 300.0), 0.0)  # as "Identify all terrain" sets it
    session._build_shapes()
    assert [len(p) for p in session._light_outlines()] == [5]  # the piece file
    settled(session, frame)
    session.dispatch("snapshot.bases")
    state = session.step(frame).state
    nob = next(i for i in range(len(state["bases"])) if game_name(session, i) == "Nob")
    click_base(session, state, nob)
    session.dispatch("measure.shoot")

    hover(session, (300.0, 150.0))  # the piece is between the pointer and Nob, next to Nob
    far = session.step(frame)
    assert_json_safe(far.state)
    assert far.state["shot"] == {"obscured": False, "in_cover": True, "cover_by": "light", "too_near": False, "intervening": 0, "shooter": None}
    hover(session, (450.0, 350.0))  # from the side: the piece is not in the lines
    side = session.step(frame)
    assert side.state["shot"]["in_cover"] is False and side.state["shot"]["too_near"] is False
    assert not np.array_equal(side.image, far.image)
    hover(session, (300.0, 285.0))  # on the piece, within 2 in of Nob
    near = session.step(frame).state["shot"]
    assert near["in_cover"] is False and near["too_near"] is True and near["cover_by"] is None

    session.dispatch("map.layout", name=None)  # no layout: no light terrain
    hover(session, (300.0, 150.0))
    assert session.step(frame).state["shot"]["in_cover"] is False


FOUND_MAP = LIGHT_MAP.replace("at_mm = [300, 300]", "at_mm = [450, 420]").replace("C5", "C1")  # C1: 81 x 63 mm on the card
PIECE_AT, PIECE_TURN = (458.0, 413.0), 188.0  # where the piece is on the board: not as on the card, in its rectangle plus the margin


PIECE_OUTLINE = np.array([[-35, -22], [10, -22], [14, -15], [35, -15], [35, 0], [20, 22], [-20, 22], [-24, 16], [-35, 16]], float)


def light_session(tmp_path, layout=True, card="C1"):
    """Layout with one light terrain piece that has a piece file, and a
    board with that piece on it, away from its card place. `layout` False:
    no layout is chosen. `card`: the key on the layout card (the piece
    file is C1)."""
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    rng = np.random.default_rng(5)
    h, w = frame.shape[:2]
    pattern = cv2.resize(rng.integers(40, 230, (h // 12 + 1, w // 12 + 1, 3)).astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    mask = np.zeros((h, w), np.uint8)
    at = (ter.Piece("C1", PIECE_OUTLINE).at(PIECE_AT, PIECE_TURN) + 80.0) * 2.0  # the frame has 2 px per mm and 80 mm of table around the board
    cv2.fillPoly(mask, [np.rint(at).astype(np.int32)], 255, cv2.LINE_AA)
    frame[mask > 127] = pattern[mask > 127]
    write_map(tmp_path / "maps", file="light.toml", text=FOUND_MAP.replace('key = "C1"', f'key = "{card}"'))
    folder = tmp_path / "terrain" / "test-world"
    folder.mkdir(parents=True)
    ter.write_piece(folder / "C1.toml", ter.Piece("C1", PIECE_OUTLINE))
    session = make_session(tmp_path, cfg, frame)
    if layout:
        session.dispatch("map.layout", name="Light")
    return session, frame


def click_mm(session, at_mm, action="board.click") -> None:
    p = np.asarray(session._to_px(np.asarray(at_mm, float).reshape(1, 2)), float).reshape(2) * session.win.scale
    session.dispatch(action, x=float(p[0]), y=float(p[1]))


@pytest.mark.parametrize("snapshot", [False, True])
def test_identify_terrains_finds_the_piece(tmp_path, snapshot):
    session, frame = light_session(tmp_path)
    before = settled(session, frame)
    assert_json_safe(before.state)
    assert before.state["light"] == [{"index": 0, "key": "C1", "center_mm": [450.0, 420.0], "angle_deg": 0.0, "located": False, "outline": True}]
    assert before.state["light_can_locate"] is True and ACTIONS_BY_ID["light.locate"].enabled(before.state)  # a layout with a piece file
    assert [s.located for s in session.terrain_shapes if s.kind == "light"] == [False]
    if snapshot:
        session.dispatch("snapshot.bases")  # a snapshot does not search the terrain
        assert session.step(frame).state["light"][0]["located"] is False

    session.dispatch("light.locate")
    found = session.step(frame)
    piece = found.state["light"][0]
    assert piece["located"] and np.linalg.norm(np.subtract(piece["center_mm"], PIECE_AT)) < 1.0
    assert abs(piece["angle_deg"] - PIECE_TURN) <= 1.0
    assert found.state["notice"]["text"] == "light terrain: 1 of 1 identified"
    shape = next(s for s in session.terrain_shapes if s.kind == "light")
    assert shape.located and shape.center == pytest.approx(PIECE_AT, abs=1.0) and len(shape.points) == len(PIECE_OUTLINE)
    session.dispatch("light.locate")  # an identified piece keeps its pose
    assert session.step(frame).state["light"][0] == piece

    assert not ACTIONS_BY_ID["light.remove"].enabled(found.state)  # no piece is selected
    if not snapshot:
        session.dispatch("view.snapshot")  # clicks need the snapshot
        session.step(frame)
    click_mm(session, PIECE_AT)
    s = session.step(frame).state
    assert s["selected_light"] == piece and s["selected"] is None
    assert ACTIONS_BY_ID["light.remove"].enabled(s) and ACTIONS_BY_ID["base.deselect"].enabled(s)
    session.dispatch("light.remove")
    s = session.step(frame).state
    assert s["light"][0]["located"] is False and s["light"][0]["center_mm"] == [450.0, 420.0]
    assert s["selected_light"] == s["light"][0] and not ACTIONS_BY_ID["light.remove"].enabled(s)  # the card symbol stays selected
    session.dispatch("light.remove")
    assert "identified light terrain piece first" in session.state()["notice"]["text"]


def test_light_terrain_not_on_the_board_stays_as_on_the_card(tmp_path):
    session, _ = light_session(tmp_path)
    bare = paint_board(ident_cfg(), GAME)  # the board without the piece
    settled(session, bare)
    session.dispatch("light.locate")
    state = session.step(bare).state
    assert state["light"][0]["located"] is False and state["light"][0]["center_mm"] == [450.0, 420.0]
    assert state["notice"]["text"] == "light terrain: 0 of 1 identified; not found: C1"


def test_identify_terrains_needs_a_layout_with_a_piece_file(tmp_path):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    write_map(tmp_path / "maps", file="light.toml", text=LIGHT_MAP)
    session = make_session(tmp_path, cfg, frame)
    state = settled(session, frame).state
    assert state["light_can_locate"] is False and not ACTIONS_BY_ID["view.light"].enabled(state)  # no layout
    session.dispatch("light.locate")
    assert "layout first" in session.state()["notice"]["text"]
    session.dispatch("map.layout", name="Light")
    state = session.step(frame).state
    assert state["light_can_locate"] is False  # no piece file
    session.dispatch("light.locate")
    assert session.state()["notice"]["text"] == "light terrain: 0 of 1 identified; no outline file: C5"


def locate_at(session, frame, at_mm):
    """"Locate terrain", then a click at `at_mm`. Clicks need the snapshot."""
    if not session.frozen:
        session.dispatch("view.snapshot")
        session.step(frame)
    session.dispatch("light.locate_at")
    assert session.state()["locating_light"] is True
    click_mm(session, at_mm)
    return session.step(frame).state


def test_locate_terrain_without_a_layout(tmp_path):
    session, frame = light_session(tmp_path, layout=False)
    before = settled(session, frame).state
    assert_json_safe(before)
    assert before["map_layout"] is None and before["light"] == [] and before["light_files"] == ["C1"]
    assert ACTIONS_BY_ID["light.locate_at"].enabled(before) and not ACTIONS_BY_ID["view.light"].enabled(before)

    s = locate_at(session, frame, (PIECE_AT[0] - 26.0, PIECE_AT[1] + 4.0))  # on the piece, far from its centre
    assert s["locating_light"] is False and s["notice"]["text"] == "light terrain C1 located"
    (piece,) = s["light"]
    assert piece["key"] == "C1" and piece["located"] and np.linalg.norm(np.subtract(piece["center_mm"], PIECE_AT)) < 1.0
    assert abs(piece["angle_deg"] - PIECE_TURN) <= 1.0
    assert [(sh.kind, sh.located) for sh in session.terrain_shapes] == [("light", True), ("light_key", False)]
    assert ACTIONS_BY_ID["view.light"].enabled(s)  # a piece to show

    locate_at(session, frame, PIECE_AT)  # the same piece again: nothing new
    assert len(session.step(frame).state["light"]) == 1

    click_mm(session, PIECE_AT)
    s = session.step(frame).state
    assert s["selected_light"] == piece and ACTIONS_BY_ID["light.remove"].enabled(s)
    session.dispatch("light.remove")  # no item of a layout: the piece goes away
    s = session.step(frame).state
    assert s["light"] == [] and s["selected_light"] is None and session.terrain_shapes == []


def test_locate_terrain_gives_the_pose_to_the_layout_item(tmp_path):
    session, frame = light_session(tmp_path)
    settled(session, frame)
    s = locate_at(session, frame, PIECE_AT)
    (piece,) = s["light"]  # the item of the layout, no second piece
    assert piece["located"] and np.linalg.norm(np.subtract(piece["center_mm"], PIECE_AT)) < 1.0
    assert session.light_poses.keys() == {0} and session.loose_light == []
    session.dispatch("map.layout", name=None)  # the poses belong to the layout
    assert session.step(frame).state["light"] == []


def test_identify_terrain_here_from_the_card_symbol(tmp_path):
    session, frame = light_session(tmp_path)
    settled(session, frame)
    session.dispatch("view.snapshot")  # clicks need the snapshot
    session.step(frame)
    click_mm(session, (450.0, 420.0))  # the card symbol
    s = session.step(frame).state
    assert s["selected_light"]["located"] is False and s["selected_light"]["index"] == 0
    assert ACTIONS_BY_ID["light.locate_here"].enabled(s) and not ACTIONS_BY_ID["light.remove"].enabled(s)
    assert session.current().crop is None  # no image of a card symbol

    session.dispatch("light.locate_here")
    found = session.step(frame)
    piece = found.state["selected_light"]  # the identified piece is selected
    assert piece["located"] and piece["index"] == 0 and np.linalg.norm(np.subtract(piece["center_mm"], PIECE_AT)) < 1.0
    assert found.state["notice"]["text"] == "light terrain C1 located" and session.loose_light == []
    assert not ACTIONS_BY_ID["light.locate_here"].enabled(found.state) and ACTIONS_BY_ID["light.remove"].enabled(found.state)
    crop = found.crop  # the piece cut from the snapshot, with board around it
    side = int(round((70.0 * abs(np.cos(np.radians(PIECE_TURN))) + 44.0 * abs(np.sin(np.radians(PIECE_TURN))) + 2 * 16.0) * session.cfg.bases_px_per_mm))
    assert crop is not None and crop.ndim == 3 and abs(crop.shape[0] - side) <= 8 and crop.shape[0] == crop.shape[1]
    assert session.current().crop is crop  # kept while the selection stays


def test_identified_piece_of_another_key_covers_the_card_symbol(tmp_path, monkeypatch):
    """A C1 piece found on the card symbol of C3 (no C3 piece file): the
    card symbol and its key are not drawn, not clicked, and give no cover."""
    session, frame = light_session(tmp_path, card="C3")
    settled(session, frame)
    session.dispatch("view.snapshot")
    session.step(frame)
    click_mm(session, (450.0, 420.0))
    assert session.step(frame).state["selected_light"]["key"] == "C3"
    session.dispatch("light.locate_here")
    s = session.step(frame).state
    assert s["selected_light"]["key"] == "C1" and s["selected_light"]["located"] and s["selected_light"]["index"] == 1  # a piece of its own, selected
    card = next(sh for sh in session.terrain_shapes if sh.kind == "light")
    key = next(sh for sh in session.terrain_shapes if sh.kind == "light_key")
    assert card.name == "C3" and card.covered and key.name == "C3" and key.covered
    assert len(session._light_outlines()) == 2 and len(session._light_outlines(cover=True)) == 1

    drawn = []
    monkeypatch.setattr(live_mod, "draw_terrain", lambda img, shapes, to_px: drawn.extend(shapes))
    session.keys = True
    session._draw_terrain(np.zeros((10, 10, 3), np.uint8), session._to_px)
    assert [(sh.kind, sh.name) for sh in drawn if sh.kind.startswith("light")] == [("light", "C1"), ("light_key", "C1")]

    session.dispatch("base.deselect")
    in_card = (428.0, 442.0)  # in the card symbol, not on the piece
    assert cv2.pointPolygonTest(card.points.astype(np.float32).reshape(-1, 1, 2), in_card, False) > 0
    click_mm(session, in_card)
    assert session.step(frame).state["selected_light"] is None
    click_mm(session, PIECE_AT)
    assert session.step(frame).state["selected_light"]["key"] == "C1"


def test_identify_terrain_here_with_no_piece_keeps_the_card_symbol(tmp_path):
    session, _ = light_session(tmp_path)
    bare = paint_board(ident_cfg(), GAME)  # the board without the piece
    settled(session, bare)
    session.dispatch("view.snapshot")
    session.step(bare)
    click_mm(session, (450.0, 420.0))
    session.step(bare)
    session.dispatch("light.locate_here")
    s = session.step(bare).state
    assert s["selected_light"]["located"] is False and s["notice"]["text"].startswith("no light terrain piece found there")


def test_locate_terrain_with_no_piece_there_changes_nothing(tmp_path, capsys):
    session, frame = light_session(tmp_path, layout=False)
    settled(session, frame)
    s = locate_at(session, frame, (150.0, 150.0))  # bare board
    assert s["light"] == [] and s["locating_light"] is False
    assert "locate terrain: no piece found at (150, 150) mm" in capsys.readouterr().out
    assert s["notice"]["text"] == "no light terrain piece found there: click the middle of the piece"


def test_locate_terrain_needs_a_clear_winner(tmp_path, capsys):
    session, frame = light_session(tmp_path, layout=False)
    ter.write_piece(tmp_path / "terrain" / "test-world" / "C2.toml", ter.Piece("C2", PIECE_OUTLINE))  # a second piece with the same outline
    session._read_pieces()
    settled(session, frame)
    s = locate_at(session, frame, PIECE_AT)
    assert s["light"] == [] and s["light_files"] == ["C1", "C2"]
    assert "locate terrain: no clear result" in capsys.readouterr().out
    assert s["notice"]["text"] == "light terrain not sure: C1 or C2"


def test_locate_terrain_shows_the_board_under_the_pointer(tmp_path):
    session, frame = light_session(tmp_path, layout=False)
    settled(session, frame)
    session.dispatch("view.snapshot")
    session.step(frame)
    session.dispatch("light.locate_at")
    assert session.current().crop is None  # no pointer on the board yet
    p = np.asarray(session._to_px(np.asarray(PIECE_AT, float).reshape(1, 2)), float).reshape(2) * session.win.scale
    session.dispatch("board.hover", x=float(p[0]), y=float(p[1]))
    crop = session.current().crop
    span = 2.0 * float(np.linalg.norm(PIECE_OUTLINE, axis=1).max())  # the largest piece file fits
    side = int(round((span + 2 * PREVIEW_PAD_MM) * session.cfg.bases_px_per_mm))
    assert crop is not None and crop.shape[:2] == (side, side)
    click_mm(session, PIECE_AT)
    s = session.step(frame)
    assert s.state["locating_light"] is False and s.state["light"][0]["located"] and s.crop is None  # the click ends it


def test_locate_terrain_stops_with_esc_and_needs_a_piece_file(tmp_path):
    session, frame = light_session(tmp_path, layout=False)
    settled(session, frame)
    session.dispatch("light.locate_at")
    assert session.state()["locating_light"] is True and not ACTIONS_BY_ID["light.locate_at"].enabled(session.state())
    session.dispatch("snapshot.back")
    assert session.state()["locating_light"] is False

    cfg = ident_cfg()
    bare = paint_board(cfg, GAME)
    write_map(tmp_path / "other" / "maps", file="light.toml", text=LIGHT_MAP)
    none = make_session(tmp_path / "other", cfg, bare)
    state = settled(none, bare).state
    assert state["light_files"] == [] and not ACTIONS_BY_ID["light.locate_at"].enabled(state)
    none.dispatch("light.locate_at")
    assert "no outline file" in none.state()["notice"]["text"] and none.state()["locating_light"] is False


def test_light_terrain_check_box_shows_and_hides_it(tmp_path):
    session, frame = light_session(tmp_path)
    shown = settled(session, frame)
    action = ACTIONS_BY_ID["view.light"]
    assert shown.state["light_shown"] is True and action.checked(shown.state) and action.enabled(shown.state)
    session.dispatch("view.light")
    hidden = session.step(frame)
    assert hidden.state["light_shown"] is False and not np.array_equal(hidden.image, shown.image)
    session.dispatch("view.terrain")  # with the layout off too, the image does not change: the light terrain was hidden
    session.dispatch("view.light")
    assert session.step(frame).state["light_shown"] is True


def test_shoot_at_draws_targeting_lines_from_the_pointer(tmp_path):
    session, frame, state, nob, burna = wall_session(tmp_path)
    action = ACTIONS_BY_ID["measure.shoot"]
    assert not action.enabled(state) and state["shooting"] is False and state["shot"] is None
    session.dispatch("measure.shoot")
    assert session.state()["shooting"] is False  # no selected base
    click_base(session, state, nob)
    before = session.step(frame)
    assert action.enabled(before.state) and before.crop is None  # bases only: no crop of the base
    session.dispatch("measure.shoot")
    s = session.state()
    assert s["shooting"] is True and action.checked(s) and s["shot"] is None  # the pointer is not on the board yet

    hover(session, GAME["Burna"][0])  # the wall is between Burna and Nob
    shot = session.step(frame)
    assert_json_safe(shot.state)
    assert shot.state["shot"] == {"obscured": True, "in_cover": False, "cover_by": None, "too_near": False, "intervening": 3, "shooter": burna}
    assert shot.aim_crop is not None and shot.crop is None and not np.array_equal(shot.image, before.image)  # the board under the pointer

    hover(session, (300.0, 250.0))  # open floor, no base
    clear = session.step(frame)
    assert clear.state["shot"] == {"obscured": False, "in_cover": False, "cover_by": None, "too_near": False, "intervening": 0, "shooter": None}
    assert not np.array_equal(clear.image, shot.image)
    hover(session, GAME["Nob"][0])  # on the target
    assert session.state()["shot"] is None

    session.dispatch("snapshot.back")  # Esc stops it, the selection stays
    s = session.state()
    assert s["shooting"] is False and s["selected"] == nob and s["mode"] == "snapshot"
    session.dispatch("measure.shoot")
    session.dispatch("base.select", index=burna)  # a new selection stops it
    assert session.state()["shooting"] is False


def test_shoot_at_an_open_hatchway_is_not_heavy_terrain(tmp_path):
    session, _, _, _, _ = wall_session(tmp_path)

    def covers_gap(rects) -> bool:
        cx, cy = session.openings[0].points.mean(axis=0)
        return bool(np.any((rects[:, 0] <= cx) & (cx <= rects[:, 2]) & (rects[:, 1] <= cy) & (cy <= rects[:, 3])))

    assert covers_gap(session._heavy_rects())  # closed
    x, y = opening_px(session)
    session.dispatch("board.click", x=x, y=y)
    session.dispatch("opening.toggle")
    assert not covers_gap(session._heavy_rects())  # open: a gap in the wall
    session.dispatch("map.layout", name=None)
    assert len(session._heavy_rects()) == 0


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


def close_session(tmp_path, other: str | None = "Grot", marks: bool = True):
    """Named snapshot of CLOSE; `other` goes to a second team. `marks`: control marks on."""
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
    if marks:
        session.dispatch("view.marks")
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


def test_control_marks_are_off_at_the_start_and_show_melee(tmp_path):
    session, frame, index = close_session(tmp_path, marks=False)
    assert ACTIONS_BY_ID["view.marks"].enabled(session.state())
    start = session.step(frame)
    assert start.state["marks"] is False and start.state["control"] is None and yellow_pixels(start.image) == 0
    session.dispatch("view.marks")
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


def test_dead_operatives_are_in_no_melee(tmp_path):
    session, frame, index = close_session(tmp_path)
    session._max_wounds = lambda label: 5  # all operatives have 5 wounds
    alive = session.step(frame)
    assert alive.state["control"]["melee"] == [sorted([index["Nob"], index["Grot"]])]
    session.wounds["elves/Grot"] = 0
    session.dirty = True
    dead = session.step(frame)
    assert dead.state["control"]["melee"] == [] and dead.state["bases"][index["Grot"]]["dead"]
    assert not dead.state["bases"][index["Nob"]]["dead"]
    assert int(np.sum(np.all(dead.image == overlay.DEAD_MARK_COLOR, axis=2))) > int(np.sum(np.all(alive.image == overlay.DEAD_MARK_COLOR, axis=2)))


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
    session.dispatch("score.skull", slot=0, category="tac", tp=3, skull=2)
    session.dispatch("score.primary", slot=0, value="crit")
    session.dispatch("score.cp", slot=0, delta=-1)
    session.dispatch("score.tac_op", slot=0, name="Rout")
    left = session.state()["scores"][0]
    assert left["cells"]["crit"] == [2, 0, 0] and left["points"] == {"crit": 2, "tac": 2, "kill": 0}
    assert left["primary_bonus"] == 1 and left["total"] == 5 and left["cp"] == 2 and left["tac_op"] == "Rout"
    session.dispatch("score.skull", slot=0, category="kill", tp=None, skull=3)  # Kill Op is automatic
    assert "automatic" in session.state()["notice"]["text"] and session.state()["scores"][0]["points"]["kill"] == 0
    assert session.state()["scores"][1]["total"] == 0

    session.dispatch("score.tac_op", slot=0, name="Flank")  # a Recon op: not for orks
    assert "not a Tac Op" in session.state()["notice"]["text"] and session.state()["scores"][0]["tac_op"] == "Rout"
    session.dispatch("score.skull", slot=0, category="crit", tp=5, skull=1)
    assert "turning point" in session.state()["notice"]["text"]
    session.dispatch("score.skull", slot=2, category="crit", tp=2, skull=1)
    assert "no team slot" in session.state()["notice"]["text"]

    session.dispatch("team.set", slot=0, name="elves")  # the score stays, the Tac Op of orks goes
    left = session.state()["scores"][0]
    assert left["total"] == 5 and left["tac_op"] is None and left["tac_ops"] == list(sc.TAC_OPS["Recon"])


def test_turn_and_crit_op_of_the_game(tmp_path):
    from battlesight import score as sc

    cfg = ident_cfg()
    session = make_session(tmp_path, cfg, paint_board(cfg, GAME))
    s = session.state()
    assert_json_safe(s)
    assert s["turn"] == 1 and s["turns"] == [1, 2, 3, 4] and s["crit_op"] is None and s["crit_ops"] == list(sc.CRIT_OPS)
    session.dispatch("game.turn", delta=-1)  # not below 1
    assert session.state()["turn"] == 1
    for _ in range(5):
        session.dispatch("game.turn", delta=1)
    assert session.state()["turn"] == 4  # not above 4
    session.dispatch("game.turn", delta=True)
    assert "whole number" in session.state()["notice"]["text"] and session.state()["turn"] == 4
    session.dispatch("game.crit_op", name="Loot")
    assert session.state()["crit_op"] == "Loot"
    session.dispatch("game.crit_op", name="Rout")  # a Tac Op
    assert "not a Crit Op" in session.state()["notice"]["text"] and session.state()["crit_op"] == "Loot"
    session.dispatch("game.crit_op", name=None)
    assert session.state()["crit_op"] is None


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
    assert [r["name"] for r in rows["operatives"]] == ["Grot #1", "Grot #2", "Nob", "Burna"]  # maximum wounds first, no value last
    assert [r["label"] for r in rows["operatives"]][1] == "orks/Grot#2"
    assert rows["missing"] == [{"label": "orks/Grot", "name": "Grot", "max": 3, "copies_in_play": 2}]
    session.dispatch("roster.add", slot=0, label="orks/Grot")
    s = session.state()
    assert [r["name"] for r in s["rosters"][0]["operatives"]] == ["Grot #1", "Grot #2", "Grot #3", "Nob", "Burna"] and s["rosters"][0]["missing"] == []
    session.dispatch("roster.add", slot=0, label="orks/Grot")
    assert "all 3 copies" in session.state()["notice"]["text"]
    session.dispatch("roster.add", slot=0, label="orks/Nob#2")
    assert "cannot be added" in session.state()["notice"]["text"]
    # Remove copy 2: its wounds go; it comes back with all wounds and the same number.
    session.dispatch("wounds.set", value=1, label="orks/Grot#2")
    session.dispatch("operative.select", label="orks/Grot#2")
    session.dispatch("roster.remove", slot=0)
    names = [r["name"] for r in session.state()["rosters"][0]["operatives"]]
    assert names == ["Grot #1", "Grot #3", "Nob", "Burna"] and "orks/Grot#2" not in session.wounds
    session.dispatch("roster.add", slot=0, label="orks/Grot")
    rows = session.state()["rosters"][0]["operatives"]
    assert rows[1]["name"] == "Grot #2" and rows[1]["wounds"] == 5
    assert session.gallery is None or session.gallery.copies["orks/Grot"] == 3


def test_copies_get_numbers_by_position_and_keep_them(grots):
    cfg, session = grots
    state = snap(session, paint_copies(cfg, [("Grot", A, 0.0), ("Grot", B, 90.0), *OTHERS]))
    assert_json_safe(state)
    grot = grot_bases(session, state)
    assert grot[A]["label"] == "orks/Grot#1" and grot[B]["label"] == "orks/Grot#2"  # first snapshot: nearest the top-left corner first
    assert grot[B]["name"] == "Grot #2" and grot[B]["copy"] == 2 and grot[B]["sure"]
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


def test_change_id_lists_each_copy_with_its_wounds(grots):
    cfg, session = grots
    state = snap(session, paint_copies(cfg, [("Grot", A, 0.0), ("Grot", B, 90.0), *OTHERS]))
    session.dispatch("wounds.set", value=3, label="orks/Grot#2")
    index = grot_bases(session, state)[A]["index"]
    session.dispatch("base.select", index=index)
    s = session.state()
    assert [c["label"] for c in s["candidates"] if c["label"].startswith("orks/Grot")] == ["orks/Grot"]  # "Change operative": no copy rows
    copies = s["copy_choices"]
    assert [(c["name"], c["copy"], c["wounds"], c["current"]) for c in copies] == [("Grot #1", 1, 5, True), ("Grot #2", 2, 3, False)]
    from battlesight.ui.parts import copy_text

    assert copy_text(copies[1]) == "Grot #2 (3/5)"
    # A type label picks the copy that the base has.
    session.dispatch("base.choose", label="orks/Grot", index=index)
    assert session.state()["bases"][index]["label"] == "orks/Grot#1"


def grot_models(session):
    """A snapshot of two Grots, plus a variant model `Grot - big` (counts as Grot).
    Returns the two Grot bases (index, live label)."""
    session.gallery.ensure_operative("orks", "Grot - big", None, counts_as="Grot")
    found = [(b["index"], b["label"]) for b in session.state()["bases"] if b["type_label"] == "orks/Grot"]
    assert len(found) == 2 and all(idt.copy_number(lab) is not None for _, lab in found)
    return found


def model_scores(session, index: int, grot: float, big: float, sure: bool = True) -> idt.Ident:
    x = session.session.idents[index]
    session.session.idents[index] = replace(x, candidates=[("orks/Grot", grot), ("orks/Grot - big", big), ("orks/Nob", 0.1)], sure=sure, confirmed=False)
    return session.session.idents[index]


def test_auto_model_switches_only_above_the_margin(grots):
    cfg, session = grots
    snap(session, paint_copies(cfg, [("Grot", A, 0.0), ("Grot", B, 90.0), *OTHERS]))
    (i, label), _ = grot_models(session)
    model_scores(session, i, 0.6, 0.7)
    session._update_models()
    assert session.models[label] == "orks/Grot - big"
    model_scores(session, i, 0.74, 0.7)  # wins by less than the margin: no change
    session._update_models()
    assert session.models[label] == "orks/Grot - big"
    model_scores(session, i, 0.95, 0.7, sure=False)  # an unsure base keeps its model
    session._update_models()
    assert session.models[label] == "orks/Grot - big"
    model_scores(session, i, 0.76, 0.7)
    session._update_models()
    assert session.models[label] == "orks/Grot"
    # The model shows in the base state and in the "Change operative" list only.
    session.dispatch("base.select", index=i)
    s = session.state()
    base = s["bases"][i]
    assert base["model"] == "orks/Grot" and base["model_name"] == "Grot" and not base["model_locked"] and base["name"].startswith("Grot #")
    rows = {c["label"]: c for c in s["candidates"] if not c["auto"]}
    assert rows["orks/Grot"]["current"] and not rows["orks/Grot - big"]["current"] and rows["orks/Grot - big"]["type_label"] == "orks/Grot"
    assert s["candidates"][-1]["auto"] and s["candidates"][-1]["current"]
    assert {b["model"] for b in s["bases"] if b["type_label"] == "orks/Nob"} == {None}  # one model only: no model


def test_chosen_model_is_locked_until_auto(grots):
    cfg, session = grots
    snap(session, paint_copies(cfg, [("Grot", A, 0.0), ("Grot", B, 90.0), *OTHERS]))
    (i, label), _ = grot_models(session)
    model_scores(session, i, 0.8, 0.6)
    session.dispatch("base.choose", label="orks/Grot - big", index=i)  # "Change operative": a model row
    s = session.state()
    assert s["bases"][i]["label"] == label and s["bases"][i]["model"] == "orks/Grot - big" and s["bases"][i]["model_locked"]
    assert any(p.parent.name == "Grot - big" for p in session.gallery.files)  # the fix photo goes to the model
    model_scores(session, i, 0.9, 0.1)
    session._update_models()
    assert session.models[label] == "orks/Grot - big"  # locked: no auto read
    session.dispatch("base.model_auto", index=i)
    assert session.models[label] == "orks/Grot" and label not in session.model_locked


def test_change_id_moves_the_models_with_the_bases(grots):
    cfg, session = grots
    snap(session, paint_copies(cfg, [("Grot", A, 0.0), ("Grot", B, 90.0), *OTHERS]))
    (i, first), (j, second) = grot_models(session)
    session.models = {first: "orks/Grot - big", second: "orks/Grot"}
    session.model_locked = {first}
    session.dispatch("base.choose", label=second, index=i)  # "Change ID": the bases swap copies
    s = session.state()
    assert s["bases"][i]["label"] == second and s["bases"][j]["label"] == first
    assert s["bases"][i]["model"] == "orks/Grot - big" and s["bases"][i]["model_locked"]  # the miniature stays on its base
    assert s["bases"][j]["model"] == "orks/Grot" and not s["bases"][j]["model_locked"]
    # A removed copy loses its model; a team change drops the models of the old team.
    session.dispatch("operative.select", label=second)
    session.dispatch("roster.remove", slot=0)
    assert second not in session.models and second not in session.model_locked
    session.dispatch("team.set", slot=0, name=None)
    assert session.models == {} and session.model_locked == set()


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
    assert near_color(patch, colors.RANGE_ZONE_COLOR).any()  # the 1 in zone of the enemy

    near = np.asarray(GAME["Burna"][0]) + (-50.0, 0.0)  # the rims are 2 mm apart there
    hover(session, near)
    melee = session.step(frame)
    assert_json_safe(melee.state)
    assert melee.state["move_melee"] == [burna] and melee.state["move"] is not None
    bx, by = np.rint((np.asarray(GAME["Burna"][0]) + cfg.rectify_margin_mm) * cfg.rectify_px_per_mm * session.win.scale).astype(int)
    sign = melee.image[by - 12 : by + 13, bx - 12 : bx + 13].reshape(-1, 3)
    assert near_color(sign, colors.MOVE_MELEE_COLOR, tol=10).sum() > near_color(live.image[by - 12 : by + 13, bx - 12 : bx + 13].reshape(-1, 3), colors.MOVE_MELEE_COLOR, tol=10).sum()  # the swords
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


# --- model preload ------------------------------------------------------------------


class SlowEmbedder(idt.ColorEmbedder):
    """A model that loads in 3 steps when the test releases it (`go`), or fails."""

    def __init__(self, fail: bool = False):
        import threading

        self.loaded = False
        self.go = threading.Event()
        self.at_step = threading.Event()
        self.fail = fail

    def load(self, input_px=0, progress=None):
        progress(0, 3, "import torch")
        progress(1, 3, "load weights")
        self.at_step.set()
        self.go.wait(5)
        if self.fail:
            raise OSError("no weights")
        progress(3, 3, "done")
        self.loaded = True


@pytest.fixture
def slow_model(monkeypatch):
    from .conftest import REAL_PRELOAD

    monkeypatch.setattr(LiveSession, "start_preload", REAL_PRELOAD)
    embedder = SlowEmbedder()
    monkeypatch.setitem(idt._EMBEDDERS, "color-hist", embedder)
    return embedder


def test_preload_shows_progress_and_blocks_naming(tmp_path, named, slow_model):
    cfg, frame = named
    session = make_session(tmp_path, cfg, frame, teams=("orks",))
    settled(session, frame)
    session.start_preload()
    assert slow_model.at_step.wait(5)
    s = session.step(frame).state
    assert_json_safe(s)
    assert s["model_ready"] is False and s["progress"] == {"text": "Loading models", "done": 1, "total": 3}
    actions = {a.id: a for a in ACTIONS}
    assert not actions["snapshot.name"].enabled(s) and not actions["base.add"].enabled(s)
    assert actions["snapshot.bases"].enabled(s) and actions["light.locate_at"].enabled(dict(s, light_files=["C1"]))
    session.dispatch("snapshot.name")  # the key works also with the button off: the core says no
    s = session.state()
    assert s["mode"] == "live" and "model is loading" in s["notice"]["text"]
    session.dispatch("base.add")
    assert not session.state()["adding"]
    slow_model.go.set()
    session._preload_thread.join(5)
    s = session.step(frame).state
    assert s["model_ready"] is True and s["progress"] is None
    session.dispatch("snapshot.name")
    assert session.state()["naming"]


def test_failed_preload_gives_a_notice_and_frees_naming(tmp_path, named, slow_model):
    cfg, frame = named
    slow_model.fail = True
    slow_model.go.set()
    session = make_session(tmp_path, cfg, frame, teams=("orks",))
    session.start_preload()
    session._preload_thread.join(5)
    s = session.step(frame).state
    assert s["model_ready"] is True and s["progress"] is None
    assert s["notice"]["text"] == "model not loaded: no weights"


def test_no_preload_for_a_loaded_model(tmp_path, named, monkeypatch):
    from .conftest import REAL_PRELOAD

    monkeypatch.setattr(LiveSession, "start_preload", REAL_PRELOAD)
    cfg, frame = named
    session = make_session(tmp_path, cfg, frame)
    session.start_preload()  # "color-hist": nothing to load
    assert session._preload_thread is None and session.state()["model_ready"] and session.state()["progress"] is None


def test_one_embedder_per_model_name():
    assert idt.make_embedder("color-hist") is idt.make_embedder("color-hist")
    assert idt.make_embedder("facebook/dinov2-small") is idt.make_embedder("facebook/dinov2-small")
    assert not idt.make_embedder("facebook/dinov2-small").loaded  # made, not loaded: no torch in the unit tests


def recorded_progress(session: LiveSession) -> list[tuple[str, int, int]]:
    """The `on_progress` calls of the session, and a check that the frame at
    the start of each long action has `busy` and the progress at 0."""
    seen: list[tuple[str, int, int]] = []
    session.on_progress = lambda p: seen.append((p["text"], p["done"], p["total"]))

    def busy(live):
        assert live.state["busy"]
        if live.state["progress"] is not None:  # "Locate operative" before its click is short: no progress
            assert live.state["progress"]["done"] == 0
            seen.append((live.state["progress"]["text"], 0, live.state["progress"]["total"]))

    session.on_busy = busy
    return seen


def test_identify_and_locate_operative_report_progress(tmp_path, named):
    cfg, frame = named
    session = make_session(tmp_path, cfg, frame, teams=("orks",))
    settled(session, frame)
    seen = recorded_progress(session)
    session.dispatch("snapshot.name")  # the gallery load, 4 stages, the name session
    assert seen[0] == ("Identifying operatives", 0, 1) and seen[-1] == ("Identifying operatives", 6, 6)
    assert [d for _, d, t in seen[1:]] == sorted(d for _, d, t in seen[1:]) and {t for _, _, t in seen[1:]} == {6}
    s = session.state()
    assert s["progress"] is None and not s["busy"] and s["naming"]
    seen.clear()
    session.dispatch("snapshot.name")  # the gallery is loaded
    assert seen[-1] == ("Identifying operatives", 5, 5)
    seen.clear()
    session.dispatch("snapshot.bases")
    assert seen[-1] == ("Finding bases", 3, 3)

    gone = session.state()["bases"][1]
    session.dispatch("snapshot.name")
    session.dispatch("base.select", index=1)
    session.dispatch("base.remove")
    session.dispatch("base.add")
    seen.clear()
    x, y = gone["position_px"]
    session.dispatch("board.click", x=x, y=y)  # the base search, then its name
    assert seen == [("Locating operative", 0, 1), ("Locating operative", 1, 2), ("Locating operative", 2, 2)]
    assert len(session.state()["bases"]) == 3 and session.state()["progress"] is None


def test_identify_and_locate_terrain_report_progress(tmp_path):
    session, frame = light_session(tmp_path)
    settled(session, frame)
    seen = recorded_progress(session)
    session.dispatch("light.locate")  # one step per piece of the layout
    assert seen == [("Identifying terrain", 0, 1), ("Identifying terrain", 0, 1), ("Identifying terrain", 1, 1)]
    session.light_poses.clear()
    session._build_shapes()
    seen.clear()
    locate_at(session, frame, PIECE_AT)  # one step per piece file
    assert seen == [("Locating terrain", 0, 1), ("Locating terrain", 1, 1)]
    assert session.state()["progress"] is None and not session.state()["busy"]


def test_shoot_at_pauses_the_measurement_and_cancel_brings_it_back(tmp_path):
    session, frame, state, nob, burna = wall_session(tmp_path)
    click_base(session, state, nob)
    session.dispatch("ring.6")
    session.dispatch("measure.move")
    session.dispatch("measure.shoot")
    s = session.state()
    assert s["shooting"] is True and s["movement"] is False and s["ring_in"] is None and s["target"] is None
    session.dispatch("snapshot.back")  # cancel: the measurement returns
    s = session.state()
    assert s["shooting"] is False and s["movement"] is True and s["ring_in"] == 6 and s["selected"] == nob
    session.dispatch("measure.shoot")
    session.dispatch("measure.shoot")  # the toggle cancels too
    assert session.state()["ring_in"] == 6
    session.dispatch("measure.shoot")
    session.dispatch("ring.3")  # a new measurement replaces the paused one
    s = session.state()
    assert s["shooting"] is False and s["ring_in"] == 3
    session.dispatch("measure.shoot")
    session.dispatch("measure.move")  # movement on: shooting off, nothing restored
    s = session.state()
    assert s["shooting"] is False and s["movement"] is True and s["ring_in"] is None


def test_shoot_at_click_locks_the_shot_and_unlock_frees_it(tmp_path):
    session, frame, state, nob, burna = wall_session(tmp_path)
    click_base(session, state, nob)
    session.dispatch("measure.shoot")
    session.dispatch("board.hover", x=300.0, y=200.0)
    at = session.hover_mm.copy()
    session.dispatch("board.click", x=300.0, y=200.0)
    s = session.state()
    assert s["shot_locked"] is True and s["shooting"] is True and s["selected"] == nob
    session.dispatch("board.hover", x=100.0, y=100.0)  # the pointer moves, the shot stays
    assert np.allclose(session.hover_mm, at)
    session.dispatch("board.click", x=100.0, y=100.0)  # a second click changes nothing
    assert np.allclose(session.hover_mm, at) and session.state()["selected"] == nob
    session.dispatch("measure.unlock")
    assert session.state()["shot_locked"] is False
    session.dispatch("board.hover", x=100.0, y=100.0)
    assert not np.allclose(session.hover_mm, at)
    session.dispatch("board.click", x=300.0, y=200.0)
    session.dispatch("snapshot.back")  # Esc stops it, locked or not
    s = session.state()
    assert s["shooting"] is False and s["shot_locked"] is False


def test_movement_click_locks_the_route_and_unlock_frees_it(tmp_path):
    session, frame, state, nob, burna = wall_session(tmp_path)
    click_base(session, state, nob)
    session.dispatch("measure.move")
    session.dispatch("board.click", x=300.0, y=200.0)  # no ring yet: a plain click
    assert session.state()["move_locked"] is False
    click_base(session, state, nob)
    session.dispatch("ring.1")
    session.dispatch("board.click", x=100.0, y=100.0)  # out of the movement area: a plain click, no lock
    assert session.state()["move_locked"] is False
    click_base(session, state, nob)
    session.dispatch("ring.6")
    near = dict(zip("xy", np.add(state["bases"][nob]["position_px"], (0.0, 20.0)).tolist()))  # on the movement area
    session.dispatch("board.hover", **near)
    at = session.hover_mm.copy()
    session.dispatch("board.click", **near)
    s = session.state()
    assert s["move_locked"] is True and s["shot_locked"] is False and s["selected"] == nob and s["movement"] is True
    route = s["move"]
    session.dispatch("board.hover", x=100.0, y=100.0)
    assert np.allclose(session.hover_mm, at) and session.state()["move"] == route
    session.dispatch("measure.unlock")
    assert session.state()["move_locked"] is False
    session.dispatch("board.hover", x=100.0, y=100.0)
    assert not np.allclose(session.hover_mm, at)
    session.dispatch("board.click", **near)
    assert session.state()["move_locked"] is True
    session.dispatch("ring.3")  # a new ring frees the lock
    assert session.state()["move_locked"] is False


def test_movement_has_no_second_base(tmp_path):
    session, frame, state, nob, burna = wall_session(tmp_path)
    click_base(session, state, nob)
    click_base(session, state, burna, "board.select")
    assert session.state()["target"] == burna
    session.dispatch("measure.move")
    s = session.state()
    assert s["target"] is None and s["distance"] is None
    click_base(session, state, burna, "board.select")
    s = session.state()
    assert s["target"] is None and s["selected"] == nob and s["movement"] is True


def test_ring_goes_up_to_12_in(tmp_path):
    session, frame, state, nob, burna = wall_session(tmp_path)
    click_base(session, state, nob)
    session.dispatch("ring.12")
    assert session.state()["ring_in"] == 12
    assert "ring.13" not in ACTIONS_BY_ID


# --- zoomed view ------------------------------------------------------------------


def test_zoomed_view_renders_only_the_region_at_the_screen_scale(tmp_path, named):
    cfg, frame = named
    session = make_session(tmp_path, cfg, frame, teams=("orks",), display_width=500)
    settled(session, frame)
    session.dispatch("snapshot.name")
    whole = session.step(frame)
    assert whole.region is None
    h, w = whole.image.shape[:2]
    x, y = whole.state["bases"][1]["position_px"]
    view = live_mod.ZoomView(4.0, (x / w - 0.125, y / h - 0.125, x / w + 0.125, y / h + 0.125), 500)
    session.set_display_view(view)
    zoomed = session.step(frame)
    (ww, wh), (rx, ry, rw, rh) = zoomed.region
    assert (ww, wh) == pytest.approx((w, h), abs=1) and zoomed.scale == whole.scale
    assert zoomed.image.shape[1] == 500 and rw == pytest.approx(w / 4) and rx == pytest.approx(x - w / 8)
    assert zoomed.state["bases"][1]["position_px"] == whole.state["bases"][1]["position_px"]  # clicks stay in window pixels
    session.dispatch("board.click", x=x, y=y)
    assert session.step(frame).state["selected"] == 1
    # sharper than the 100% image made bigger: more detail in the same region
    small = whole.image[int(ry) : int(ry + rh), int(rx) : int(rx + rw)]
    assert cv2.Laplacian(zoomed.image, cv2.CV_64F).var() > cv2.Laplacian(cv2.resize(small, zoomed.image.shape[1::-1]), cv2.CV_64F).var()
    session.dispatch("view.live")  # live top-down and camera view
    assert session.step(frame).image.shape[1] == 500
    session.dispatch("view.top_down")
    camera = session.step(frame)
    assert camera.image.shape[1] == 500 and camera.region[0] == pytest.approx((frame.shape[1] * camera.scale, frame.shape[0] * camera.scale))
    session.set_display_view(None)
    assert session.step(frame).region is None


def test_style_scale_keeps_lines_the_same_size_on_the_screen():
    img = np.zeros((100, 4000, 3), np.uint8)
    assert overlay._scale(img) == pytest.approx(4000 / 1920)
    with overlay.style_scale(0.7):
        assert overlay._scale(img) == 0.7
    assert overlay._scale(img) == pytest.approx(4000 / 1920)


def test_moved_operatives_show_their_old_place_until_the_names_are_confirmed(tmp_path, named, monkeypatch):
    cfg, frame = named
    calls = []
    real = live_mod.draw_move_trail_mm
    monkeypatch.setattr(live_mod, "draw_move_trail_mm", lambda img, a, b, d, to_px, color: (calls.append((tuple(np.round(a)), tuple(np.round(b)), d, color)), real(img, a, b, d, to_px, color)))
    train = paint_board(cfg, TRAIN)
    session = make_session(tmp_path, cfg, train, teams=("orks",))
    settled(session, train)
    session.dispatch("snapshot.name")
    session.step(train)
    assert session.moved_from == {} and not calls  # the first identification has no old places

    # Two operatives move, the Grot stays.
    game = paint_board(cfg, {**TRAIN, "Nob": GAME["Nob"], "Burna": GAME["Burna"]})
    settled(session, game)
    session.dispatch("snapshot.name")
    calls.clear()
    s = session.step(game).state
    moved = {idt.label_of("orks", n) for n in ("Nob", "Burna")}
    assert set(session.moved_from) == moved
    assert {(a, b) for a, b, _, _ in calls} == {(TRAIN[n][0], GAME[n][0]) for n in ("Nob", "Burna")}
    assert all(c == colors.SELECTED_COLOR for *_, c in calls)
    assert {d for _, _, d, _ in calls} == {SIZES["Nob"], SIZES["Burna"]}

    # A name chosen for one base ends its arrow; "Accept all identification" ends the rest.
    nob = next(i for i, b in enumerate(s["bases"]) if b["label"] == idt.label_of("orks", "Nob"))
    session.dispatch("base.choose", label=idt.label_of("orks", "Nob"), index=nob)
    calls.clear()
    session.step(game)
    assert {b for _, b, _, _ in calls} == {GAME["Burna"][0]}
    session.dispatch("names.accept")
    calls.clear()
    session.step(game)
    assert not calls
