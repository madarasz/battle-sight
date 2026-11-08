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
from .test_ident import enroll_from, ident_cfg, make_gallery, paint_board

TRAIN = {"Nob": ((150.0, 150.0), 0.0), "Grot": ((300.0, 200.0), 40.0), "Burna": ((450.0, 500.0), 90.0)}
GAME = {"Nob": ((300.0, 350.0), 120.0), "Grot": ((150.0, 500.0), 200.0), "Burna": ((450.0, 200.0), 300.0)}
SETTLE = 7  # more than settle_frames of the test config


def make_session(tmp_path, cfg, frame, **over) -> LiveSession:
    size = (frame.shape[1], frame.shape[0])
    args = dict(gallery_root=tmp_path / "gallery", reference_path=tmp_path / "ref" / "empty.png", snapshot_dir=tmp_path / "snaps")
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


def test_view_grid_and_tag_actions_change_state_and_image(tmp_path):
    cfg = ident_cfg()
    frame, _ = render_board(cfg)
    session = make_session(tmp_path, cfg, frame)
    session.dispatch("view.top_down")  # the camera view
    camera = settled(session, frame)

    session.dispatch("view.tags")
    no_tags = session.step(frame)
    assert no_tags.state["show_tags"] is False and not np.array_equal(no_tags.image, camera.image)

    session.dispatch("view.grid")
    no_grid = session.step(frame)
    assert no_grid.state["grid"] is False and not np.array_equal(no_grid.image, no_tags.image)

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
    session.dispatch("board.click", x=10, y=10)
    assert session.state()["notice"]["text"] == "push space to name the operatives"
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
    settled(session, frame)
    session.dispatch("snapshot.name")
    base = session.step(frame).state["bases"][0]
    session.dispatch("board.click", x=base["position_px"][0], y=base["position_px"][1])
    s = session.state()
    assert s["selected"] is None and "mm base in the gallery" in s["notice"]["text"]


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
