"""Game core of the live app (`live.LiveSession`) on synthetic boards. No camera,
no window, no model download."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

import battlesight.app as app
from battlesight import ident as idt
from battlesight.camera import CameraSettings, ImageSource
from battlesight.live import ACTIONS, LiveSession

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
    assert live.crop is session.session.crops[1]
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


def test_back_clears_selection_then_closes(tmp_path, named):
    cfg, frame = named
    session = make_session(tmp_path, cfg, frame, teams=("orks",))
    settled(session, frame)
    session.dispatch("snapshot.name")
    base = session.step(frame).state["bases"][0]
    session.dispatch("board.click", x=base["position_px"][0], y=base["position_px"][1])
    assert session.state()["selected"] == 0
    session.dispatch("snapshot.back")
    s = session.state()
    assert s["selected"] is None and s["mode"] == "snapshot"
    session.dispatch("snapshot.back")
    assert session.state()["mode"] == "live"


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


def test_zoom_view_keeps_the_anchor_and_stays_in_the_frame():
    from battlesight.live import ZoomView

    frame = np.zeros((1000, 2000, 3), np.uint8)
    z = ZoomView(1600)
    assert z.image(frame) is frame and z.area() is None and z.to_frame(5.0, 6.0) == (5.0, 6.0)
    anchor = (500.0, 250.0)
    z.step(1, anchor)
    z.step(1, anchor)
    assert z.factor == pytest.approx(2.0)
    x0, y0, x1, y1 = z.area()
    assert (x1 - x0, y1 - y0) == (1000, 500)
    shown = z.image(frame)
    assert shown.shape[:2] == (800, 1600)
    # The anchor is at the same relative place in the shown image as in the frame.
    assert z.to_frame(0.25 * 1600, 0.25 * 800) == pytest.approx(anchor, abs=1.0)
    for _ in range(20):
        z.step(1, (1999.0, 999.0))
    x0, y0, x1, y1 = z.area()
    assert z.factor == pytest.approx(ZoomView.MAX) and x1 <= 2000 and y1 <= 1000 and x0 >= 0 and y0 >= 0
    z.reset()
    assert not z.active and z.area() is None and z.image(frame) is frame


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
    click_base(session, state, 1)
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
    click_base(session, state, 1)
    click_base(session, state, 2)
    assert session.state()["target"] == 2
    click_base(session, state, 2)
    s = session.state()
    assert s["target"] is None and s["selected"] == 0 and s["distance"] is None
    session.dispatch("ring.4")
    click_base(session, state, 0)
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
    click_base(session, state, 0)
    click_base(session, state, 1)
    session.dispatch("ring.2")
    click_base(session, state, 2, "board.select")
    s = session.state()
    assert s["selected"] == 2 and s["target"] is None and s["ring_in"] is None and s["candidates"]


def test_view_change_keeps_ring_and_distance(tmp_path):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    session, state = bases_snapshot(tmp_path, cfg, frame)
    click_base(session, state, 0)
    click_base(session, state, 1)
    session.dispatch("ring.2")
    top = session.step(frame)
    session.dispatch("view.top_down")
    camera = session.step(frame)
    assert camera.state["top_down"] is False and camera.image.shape != top.image.shape
    assert camera.state["ring_in"] == 2 and camera.state["distance"] == top.state["distance"]


def test_esc_and_a_new_snapshot_clear_the_measurement(tmp_path):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    session, state = bases_snapshot(tmp_path, cfg, frame)
    click_base(session, state, 0)
    click_base(session, state, 1)
    session.dispatch("ring.2")
    session.dispatch("snapshot.back")
    s = session.state()
    assert (s["selected"], s["target"], s["ring_in"], s["distance"], s["mode"]) == (None, None, None, None, "snapshot")
    click_base(session, state, 0)
    session.dispatch("snapshot.bases")
    assert session.state()["selected"] is None


def test_choosing_a_name_clears_the_measurement(tmp_path, named):
    cfg, frame = named
    session = make_session(tmp_path, cfg, frame, teams=("orks",))
    settled(session, frame)
    session.dispatch("snapshot.name")
    state = session.step(frame).state
    click_base(session, state, 0)
    click_base(session, state, 1)
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
