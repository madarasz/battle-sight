"""The Qt front end of the live app, offscreen. No camera, no display."""

from __future__ import annotations

import ast
import os
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QMouseEvent, QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QAbstractButton, QAbstractItemView, QApplication, QGroupBox, QRadioButton  # noqa: E402

import battlesight.app as app  # noqa: E402
from battlesight.live import ACTIONS, LiveSession  # noqa: E402
from battlesight.ui import shell  # noqa: E402
from battlesight.ui.layouts import DEFAULT_LAYOUT, LAYOUTS  # noqa: E402
from battlesight.ui.layouts.inspector import InspectorLayout, page_for, summary_text  # noqa: E402
from battlesight.ui.layouts.panel import PanelLayout  # noqa: E402
from battlesight.ui.parts import debug_text, distance_label  # noqa: E402
from battlesight.colors import MODE_LIVE_COLOR, MODE_MINIMAP_COLOR, MODE_SNAPSHOT_COLOR, TEAM_1_COLOR, TEAM_2_COLOR, qt_hex  # noqa: E402
from battlesight.ui.parts import COLORS, VideoWidget  # noqa: E402
from battlesight.ui.shell import MainWindow, help_text, status_text  # noqa: E402

from .test_ident import enroll_from, ident_cfg, make_gallery, paint_board  # noqa: E402
from .test_live import GAME, TRAIN, make_session, named, named_template, settled  # noqa: E402,F401  (fixtures)


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication(["test"])


class FakeCamera:
    def __init__(self, frame):
        self.frame = frame
        self.size = (frame.shape[1], frame.shape[0])
        self.released = False

    def read(self):
        time.sleep(0.005)
        return self.frame

    def release(self):
        self.released = True


def wait_until(check, seconds: float = 20.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        QApplication.processEvents()
        value = check()
        if value:
            return value
        time.sleep(0.005)
    raise AssertionError("condition not met in time")




@pytest.fixture
def window(qapp, tmp_path, named):
    cfg, frame = named
    win = started_window(tmp_path, cfg, frame, "panel")
    yield win
    win.close()


@pytest.fixture
def inspector(qapp, tmp_path, named):
    cfg, frame = named
    win = started_window(tmp_path, cfg, frame, "inspector")
    yield win
    win.close()


def with_layout(session, name: str):
    """The session says which layout the window starts with (a restored state does it)."""
    session.ui_layout = name
    return session


def started_window(tmp_path, cfg, frame, layout: str) -> MainWindow:
    win = MainWindow(with_layout(make_session(tmp_path, cfg, frame, teams=("orks",)), layout), FakeCamera(frame))
    win.resize(1200, 900)
    win.show()
    win.start()
    wait_until(lambda: win.state is not None and win.state["drift"]["ok"])
    return win


def press(win: MainWindow, action_id: str) -> None:
    win.shortcuts[action_id].activated.emit()


def mouse_click(video: VideoWidget, pos: QPointF, button=Qt.MouseButton.LeftButton) -> None:
    global_pos = QPointF(video.mapToGlobal(pos.toPoint()))
    video.mousePressEvent(QMouseEvent(QEvent.Type.MouseButtonPress, pos, global_pos, button, button, Qt.KeyboardModifier.NoModifier))


def widget_pos(win: MainWindow, image_xy) -> QPointF:
    rect = win.video.image_rect()
    scale = rect.width() / win.video._image.width()
    return QPointF(rect.x() + image_xy[0] * scale, rect.y() + image_xy[1] * scale)


def open_names(win: MainWindow) -> dict:
    press(win, "snapshot.name")
    return wait_until(lambda: win.state if win.state["naming"] and not win.state["busy"] else None)


# --- pure functions -----------------------------------------------------------------


def base_state(**over) -> dict:
    drift = {"status": "OK", "ok": True, "text": "4 corners OK", "lost": [], "card": None}
    state = dict(camera=True, drift=drift, lens="nominal", resolution=[3840, 2160], busy=False, mode="live", reference_set=True, snapshot_ok=None, naming=False, bases=[], ring_in=None, distance=None)
    state.update(over)
    return state


@pytest.mark.parametrize(
    "over, text, level",
    [
        (dict(camera=False), "camera down", "error"),
        (dict(drift=None), "waiting for the first frame", "warn"),
        (dict(), "4 corners OK  |  nominal lens  |  3840x2160", "ok"),
        (dict(busy=True), "working", "ok"),
        (dict(mode="snapshot", snapshot_ok=False), "snapshot failed", "ok"),
        (dict(mode="snapshot", snapshot_ok=True, naming=True, bases=[{"label": "a"}, {"label": None}]), "1/2 named", "ok"),
        (dict(mode="snapshot", snapshot_ok=True, bases=[{"label": None}]), "1 bases", "ok"),
        (dict(mode="snapshot", snapshot_ok=True, bases=[{"label": None}], ring_in=6), "ring 6 in", "ok"),
        (dict(mode="snapshot", snapshot_ok=True, bases=[{"label": None}], distance={"mm": 127.0, "in": 5.0}), 'rim to rim 5.00"', "ok"),
        (dict(mode="snapshot", snapshot_ok=True, bases=[{"label": None}], distance={"mm": 127.0, "in": 5.0, "around": True}), 'around walls 5.00"', "ok"),
        (dict(mode="snapshot", snapshot_ok=True, bases=[{"label": None}], distance={"mm": None, "in": None, "around": True}), "around walls no path", "ok"),
        (dict(mode="snapshot", snapshot_ok=True, bases=[{"label": None}], ring_in=6, move={"in": 5, "increments": [3, 2]}), 'move 5" (3 + 2)', "ok"),
        (dict(mode="snapshot", snapshot_ok=True, bases=[{"label": None}], ring_in=6, move={"in": 3, "increments": [3]}), 'ring 6 in  |  move 3"', "ok"),
    ],
)
def test_status_text(over, text, level):
    got, got_level = status_text(base_state(**over))
    assert text in got and got_level == level


@pytest.mark.parametrize("status, level", [("OK", "ok"), ("NO_REFERENCE", "warn"), ("MOVED", "warn"), ("CARD_MOVED", "error"), ("LOST", "error")])
def test_status_level_of_each_drift_value(status, level):
    drift = {"status": status, "ok": False, "text": "x", "lost": [], "card": None}
    assert status_text(base_state(drift=drift))[1] == level


@pytest.mark.parametrize(
    "over",
    [
        dict(),
        dict(reference_set=False),
        dict(mode="snapshot", snapshot_ok=False),
        dict(mode="snapshot", snapshot_ok=True, naming=True, bases=[{"label": "a"}]),
        dict(mode="snapshot", snapshot_ok=True, bases=[{"label": None}]),
    ],
)
def test_status_has_no_key_hints(over):
    text = status_text(base_state(**over))[0]
    assert not any(word in text for word in ("space", "Esc", "click", "b:", "e:"))


def test_debug_text_with_and_without_values():
    full = debug_text({"debug": {"tags": 4, "fit_max_mm": 0.123, "fps": 29.97, "detect_ms": 180.4}})
    assert "4" in full and "0.12 mm" in full and "30.0 per s" in full and "180 ms" in full
    assert debug_text({"debug": {"tags": None, "fit_max_mm": None, "fps": None, "detect_ms": None}}).count("-") == 4


def test_help_text_lists_each_key():
    text = help_text()
    assert all(a.key in text for a in ACTIONS if a.key) and "click" in text


# --- video widget -------------------------------------------------------------------


def wheel(video: VideoWidget, pos: QPointF, notches: float) -> None:
    video.wheelEvent(QWheelEvent(pos, QPointF(video.mapToGlobal(pos.toPoint())), QPoint(0, 0), QPoint(0, int(120 * notches)), Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False))


def test_video_widget_zoom_and_pan(qapp):
    video = VideoWidget(zoomable=True)
    video.resize(400, 400)
    video.set_image(np.zeros((100, 200, 3), np.uint8))
    views, left = [], []
    video.view_changed.connect(views.append)
    video.clicked.connect(lambda x, y, pos: left.append((x, y)))
    assert video.zoom == 1.0 and video._level.isHidden() and not video._out.isEnabled()  # no percentage at 100%
    point = QPointF(300, 200)
    before = video.to_image(point)
    wheel(video, point, 2)
    assert video.zoom == pytest.approx(1.25**2) and not video._level.isHidden() and video._level.text() == "156%"
    assert video.to_image(point) == pytest.approx(before)  # the point under the pointer stays
    view = views[-1]  # the core gets the visible region at the screen width
    assert view.zoom == video.zoom and view.width == 400 and view.region[2] - view.region[0] == pytest.approx(1 / video.zoom)
    wheel(video, point, 50)
    assert video.zoom == 8.0 and not video._in.isEnabled()
    rect = video.image_rect()
    assert rect.left() <= 0 and rect.right() >= 400 and rect.top() <= 0 and rect.bottom() >= 400  # no gap at an edge
    # the middle button drags, and a left click still clicks
    start = video.to_image(QPointF(200, 200))
    pos = QPointF(200, 200)
    global_pos = QPointF(video.mapToGlobal(pos.toPoint()))
    middle = Qt.MouseButton.MiddleButton
    video.mousePressEvent(QMouseEvent(QEvent.Type.MouseButtonPress, pos, global_pos, middle, middle, Qt.KeyboardModifier.NoModifier))
    video.mouseMoveEvent(QMouseEvent(QEvent.Type.MouseMove, pos + QPointF(40, 0), global_pos, Qt.MouseButton.NoButton, middle, Qt.KeyboardModifier.NoModifier))
    video.mouseReleaseEvent(QMouseEvent(QEvent.Type.MouseButtonRelease, pos + QPointF(40, 0), global_pos, middle, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier))
    assert video.to_image(QPointF(200, 200))[0] < start[0] and left == []
    mouse_click(video, QPointF(200, 200))
    assert len(left) == 1
    video._out.click()
    assert video.zoom == pytest.approx(8 / 1.5)
    video._level.click()
    assert video.zoom == 1.0 and video.image_rect().width() == 400 and views[-1] is None


def test_video_widget_shows_a_region_and_clicks_in_whole_image_pixels(qapp):
    video = VideoWidget(zoomable=True)
    video.resize(480, 240)  # not below the minimum size
    video.set_image(np.zeros((100, 200, 3), np.uint8))  # the whole image at 100%
    video.zoom_by(2.0)
    region = video.zoom_view().region
    assert region == pytest.approx((0.25, 0.25, 0.75, 0.75))
    part = np.zeros((240, 480, 3), np.uint8)  # the core sends the middle part, drawn at the screen scale
    video.set_image(part, ((200.0, 100.0), (50.0, 25.0, 100.0, 50.0)))
    assert video._image.width() == 480 and video.image_rect().width() == 960  # the whole image stays the reference
    left = []
    video.clicked.connect(lambda x, y, pos: left.append((x, y)))
    mouse_click(video, QPointF(240, 120))  # the middle of the widget
    assert left == [pytest.approx((100, 50))]  # pixels of the whole image at 100%, as without zoom
    video.grab()  # paints the backdrop and the region
    crop = VideoWidget()  # a crop has no zoom
    crop.resize(400, 400)
    crop.set_image(np.zeros((100, 200, 3), np.uint8))
    wheel(crop, QPointF(200, 200), 3)
    assert crop.zoom == 1.0 and crop._bar is None


def test_video_widget_maps_clicks_with_letterbox(qapp):
    video = VideoWidget()
    video.resize(400, 400)
    video.set_image(np.zeros((100, 200, 3), np.uint8))  # 2:1 in a square: bars above and below
    rect = video.image_rect()
    assert (rect.width(), rect.height(), rect.y()) == (400, 200, 100)
    assert video.to_image(QPointF(200, 200)) == pytest.approx((100, 50))
    assert video.to_image(QPointF(200, 50)) is None  # on the bar
    left, right = [], []
    video.clicked.connect(lambda x, y, pos: left.append((x, y)))
    video.right_clicked.connect(lambda x, y, pos: right.append((x, y)))
    mouse_click(video, QPointF(100, 150))
    assert left == [pytest.approx((50, 25))] and right == []
    mouse_click(video, QPointF(300, 250), Qt.MouseButton.RightButton)
    assert len(left) == 1 and right == [pytest.approx((150, 75))]


# --- the window ---------------------------------------------------------------------


def test_window_has_keys_and_menu_for_the_actions(window):
    assert set(window.shortcuts) == {a.id for a in ACTIONS if a.key}
    assert set(window.menu_actions) == {a.id for a in ACTIONS if not a.args}
    assert window.status_label.text().startswith("4 corners OK")
    assert window.video.focusPolicy() is Qt.FocusPolicy.NoFocus


def test_keys_change_state_and_menu_follows(window):
    grid = window.state["grid"]
    press(window, "view.grid")
    wait_until(lambda: window.state["grid"] is (not grid))
    wait_until(lambda: window.menu_actions["view.grid"][1].isChecked() is (not grid))
    assert not window.menu_actions["names.undo"][1].isEnabled()


def test_menu_item_runs_the_action(window):
    window.menu_actions["view.top_down"][1].trigger()
    wait_until(lambda: window.state["top_down"] is False)


def test_snapshot_shows_busy_then_result_and_esc_goes_back(window):
    seen = []
    window.worker.frame_ready.connect(lambda live, _done: seen.append(live.state["busy"]))
    state = open_names(window)
    assert True in seen and state["mode"] == "snapshot" and len(state["bases"]) == 3
    assert "named" in window.status_label.text()
    press(window, "snapshot.back")
    wait_until(lambda: window.state["mode"] == "live")


def test_right_click_on_a_base_opens_names_and_choice_sets_it(window):
    state = open_names(window)
    base = state["bases"][1]
    mouse_click(window.video, widget_pos(window, base["position_px"]), Qt.MouseButton.RightButton)
    popup = wait_until(lambda: window.popup)
    assert window.state["selected"] == 1
    assert [a.text() for a in popup.actions()] == ["Change operative"]  # no "Change ID": one copy of each type
    action = popup.actions()[0].menu().actions()[0]
    label = window.state["candidates"][0]["label"]
    assert action.text().startswith(label)
    action.trigger()
    popup.close()
    wait_until(lambda: window.state["bases"][1]["confirmed"])
    assert window.state["bases"][1]["label"] == label and window.state["selected"] == 1  # the named base stays selected


def test_closing_the_popup_keeps_the_selection(window):
    state = open_names(window)
    base = state["bases"][0]
    mouse_click(window.video, widget_pos(window, base["position_px"]), Qt.MouseButton.RightButton)
    popup = wait_until(lambda: window.popup)
    wait_until(lambda: window.state["selected"] == 0)
    popup.close()
    done = window.worker.done
    wait_until(lambda: window.worker.done == done and window.state["selected"] == 0)
    assert window.state["mode"] == "snapshot"


def test_left_click_selects_with_no_popup(window):
    state = open_names(window)
    mouse_click(window.video, widget_pos(window, state["bases"][0]["position_px"]))
    wait_until(lambda: window.state["selected"] == 0)
    assert window.popup is None and window._pending_click is None


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_ring_key_and_second_base_show_in_the_status_area_of_each_layout(qapp, tmp_path, named, layout):
    cfg, frame = named
    win = started_window(tmp_path, cfg, frame, layout)
    try:
        state = open_names(win)
        mouse_click(win.video, widget_pos(win, state["bases"][0]["position_px"]))
        wait_until(lambda: win.state["selected"] == 0)
        press(win, "ring.6")
        wait_until(lambda: win.state["ring_in"] == 6 and "ring 6 in" in win.status_label.text())
        mouse_click(win.video, widget_pos(win, state["bases"][1]["position_px"]), Qt.MouseButton.RightButton)
        wait_until(lambda: win.state["target"] == 1 and "rim to rim" in win.status_label.text())
        wait_until(lambda: win._pending_click is None)
        assert win.popup is None  # a right click that measures opens no name popup
        press(win, "ring.clear")
        wait_until(lambda: win.state["ring_in"] is None and "ring" not in win.status_label.text().replace("rim to rim", ""))
    finally:
        win.close()


def test_click_on_empty_board_opens_no_popup(window):
    open_names(window)
    mouse_click(window.video, QPointF(window.video.width() / 2, 4 + window.video.image_rect().y()), Qt.MouseButton.RightButton)
    wait_until(lambda: window._pending_click is None)
    assert window.popup is None


def test_click_positions_stay_right_after_a_resize(window):
    state = open_names(window)
    window.resize(700, 500)
    # The image gets smaller when the worker sees the new width.
    wait_until(lambda: window.video._image.width() <= window.video.width() * window.video.devicePixelRatioF() and not window.state["busy"])
    state = wait_until(lambda: window.state if window.state["bases"][2]["position_px"] != state["bases"][2]["position_px"] else None)
    mouse_click(window.video, widget_pos(window, state["bases"][2]["position_px"]))
    wait_until(lambda: window.state["selected"] == 2)


def test_commands_are_dropped_while_busy(window):
    sent = window.worker.sent
    window.state = dict(window.state, busy=True)  # as the frame sent at the start of a snapshot
    press(window, "snapshot.name")
    press(window, "view.grid")
    mouse_click(window.video, QPointF(window.video.width() / 2, window.video.height() / 2))
    mouse_click(window.video, QPointF(window.video.width() / 2, window.video.height() / 2), Qt.MouseButton.RightButton)
    assert window.worker.sent == sent and window._pending_click is None


def test_a_bad_command_does_not_stop_the_video(window):
    window.worker.submit("no.such.action")
    wait_until(lambda: window.state["notice"] is not None and "failed" in window.state["notice"]["text"])
    grid = window.state["grid"]
    press(window, "view.grid")
    wait_until(lambda: window.state["grid"] is (not grid))


def test_camera_down_shows_status_and_comes_back(window):
    frame, window.camera.frame = window.camera.frame, None
    wait_until(lambda: window.state["camera"] is False)
    assert "camera down" in window.status_label.text()
    window.camera.frame = frame
    wait_until(lambda: window.state["camera"] is True)


def test_quit_stops_worker_and_releases_camera(qapp, tmp_path):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    win = MainWindow(make_session(tmp_path, cfg, frame), FakeCamera(frame))
    win.show()
    win.start()
    wait_until(lambda: win.state is not None)
    press(win, "app.quit")
    assert win.worker.isFinished() and win.camera.released


# --- the layout contract ------------------------------------------------------------


FORBIDDEN_IMPORTS = {"pipeline", "ident", "calib", "camera"}
LAYOUT_DIR = Path(shell.__file__).parent / "layouts"


def forbidden_imports(source: str) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [part for alias in node.names for part in alias.name.split(".")]
        elif isinstance(node, ast.ImportFrom):
            names = (node.module or "").split(".") + [alias.name for alias in node.names]
        else:
            continue
        found |= FORBIDDEN_IMPORTS & set(names)
    return found


@pytest.mark.parametrize(
    "source, found",
    [("from ...pipeline import Pipeline", {"pipeline"}), ("from ... import ident", {"ident"}), ("import battlesight.calib", {"calib"}), ("from ...live import ACTIONS", set())],
)
def test_import_check_finds_forbidden_modules(source, found):
    assert forbidden_imports(source) == found


@pytest.mark.parametrize("path", sorted(LAYOUT_DIR.glob("*.py")), ids=lambda p: p.name)
def test_layout_modules_do_not_import_the_vision_code(path):
    assert forbidden_imports(path.read_text()) == set()


@pytest.fixture
def recorded(tmp_path, named):
    """Session and frames for each state of the done test of docs/specs/ui.md, section 11."""
    cfg, frame = named
    session = make_session(tmp_path, cfg, frame, teams=("orks",))
    frames = {"before the first frame": session.current()}
    live = settled(session, frame)
    frames["live"] = live
    frames["busy"] = replace(live, state=dict(live.state, busy=True))
    frames["loading model"] = replace(live, state=dict(live.state, model_ready=False, progress={"text": "Loading models", "done": 2, "total": 5}))
    frames["camera down"] = session.step(None)
    for status in ("OK", "NO_REFERENCE", "MOVED", "CARD_MOVED", "LOST"):
        drift = {"status": status, "ok": status == "OK", "text": status, "lost": [], "card": None}
        frames[f"drift {status}"] = replace(live, state=dict(live.state, drift=drift))
    session.dispatch("snapshot.name")
    snapshot = session.step(frame)
    frames["snapshot"] = snapshot
    x, y = snapshot.state["bases"][0]["position_px"]
    session.dispatch("board.click", x=x, y=y)
    frames["base selected"] = session.step(frame)
    assert frames["base selected"].state["selected"] == 0 and frames["snapshot"].state["naming"]
    session.dispatch("ring.6")
    frames["ring"] = session.step(frame)
    x, y = snapshot.state["bases"][1]["position_px"]
    session.dispatch("board.select", x=x, y=y)
    frames["target"] = session.step(frame)
    assert frames["target"].state["distance"] is not None and frames["ring"].state["ring_in"] == 6
    session.dispatch("snapshot.bases")
    bases_only = session.step(frame)
    x, y = bases_only.state["bases"][0]["position_px"]
    session.dispatch("board.click", x=x, y=y)
    frames["base selected, no names"] = session.step(frame)
    assert frames["base selected, no names"].crop is None
    return session, frame, frames


@pytest.fixture(params=sorted(LAYOUTS))
def idle_window(request, qapp, tmp_path, recorded):
    """A window of one layout, with no worker running."""
    session, frame, _ = recorded
    win = MainWindow(with_layout(session, request.param), FakeCamera(frame))
    win.resize(1000, 700)
    win.show()
    yield win
    win.close()


@pytest.fixture
def idle_panel(qapp, tmp_path, recorded):
    session, frame, _ = recorded
    win = MainWindow(with_layout(session, "panel"), FakeCamera(frame))
    yield win
    win.close()


@pytest.fixture
def idle_inspector(qapp, tmp_path, recorded):
    session, frame, _ = recorded
    win = MainWindow(with_layout(session, "inspector"), FakeCamera(frame))
    win.show()
    yield win
    win.close()


def test_layout_builds_with_the_video_inside(idle_window):
    assert idle_window.current_layout.name in LAYOUTS
    assert idle_window.centralWidget().isAncestorOf(idle_window.video)


def test_layout_shows_each_recorded_state(idle_window, recorded):
    for frame in recorded[2].values():
        idle_window.worker._gui_free.clear()
        idle_window._on_frame(frame, 0)
        assert idle_window.worker._gui_free.is_set()


def test_distance_label_names_the_kind_of_measurement():
    assert distance_label({"mm": 127.0, "in": 5.0, "around": False}) == 'Rim to rim: 5.00"'
    assert distance_label({"mm": 127.0, "in": 5.0, "around": True}) == 'Around walls: 5.00"'
    assert distance_label({"mm": None, "in": None, "around": True}) == "Around walls: no path"


def test_each_layout_has_the_around_walls_check_box(idle_window, recorded):
    boxes = [b for b in idle_window.centralWidget().findChildren(QAbstractButton) if b.text() == "Measure around walls"]
    assert len(boxes) == 1 and boxes[0].isCheckable()
    state = dict(recorded[2]["live"].state)
    idle_window.current_layout.on_state({**state, "map_layout": None, "around_walls": False})
    assert not boxes[0].isEnabled() and not boxes[0].isChecked()
    idle_window.current_layout.on_state({**state, "map_layout": "Small", "around_walls": True})
    assert boxes[0].isEnabled() and boxes[0].isChecked()


def test_view_radio_buttons_in_the_bottom_row_of_view(idle_window, recorded):
    layout = idle_window.current_layout
    found = idle_window.centralWidget().findChildren(QRadioButton)
    assert [c.text() for c in found] == ["Live", "Snapshot", "Minimap"]
    radios = dict(zip(("view.live", "view.snapshot", "view.minimap"), found))
    assert not any(b.text() in ("Go back to Live",) for b in idle_window.centralWidget().findChildren(QAbstractButton))
    live = dict(recorded[2]["live"].state)
    layout.on_state(live)
    assert radios["view.live"].isChecked() and radios["view.snapshot"].isEnabled() and not radios["view.minimap"].isChecked()
    layout.on_state({**live, "resolution": None, "snapshot_ok": None})  # no camera frame, no snapshot
    assert not radios["view.snapshot"].isEnabled()
    layout.on_state(dict(recorded[2]["snapshot"].state, map_layout="Small", minimap=True))
    assert radios["view.minimap"].isChecked() and not radios["view.live"].isChecked() and radios["view.snapshot"].isEnabled()
    layout.on_state(dict(recorded[2]["snapshot"].state))
    assert radios["view.snapshot"].isChecked() and not radios["view.minimap"].isChecked()


def test_avoid_control_range_check_box_needs_measure_movement(idle_window, recorded):
    boxes = [b for b in idle_window.centralWidget().findChildren(QAbstractButton) if b.text() == "Avoid control range"]
    assert len(boxes) == 1 and boxes[0].isCheckable()
    state = dict(recorded[2]["base selected"].state)
    idle_window.current_layout.on_state({**state, "movement": False, "avoid_melee": False})
    assert not boxes[0].isEnabled() and not boxes[0].isChecked()
    idle_window.current_layout.on_state({**state, "movement": True, "avoid_melee": True})
    assert boxes[0].isEnabled() and boxes[0].isChecked()


def test_each_layout_has_the_movement_check_box(idle_window, recorded):
    boxes = [b for b in idle_window.centralWidget().findChildren(QAbstractButton) if b.text() == "Measure movement"]
    assert len(boxes) == 1 and boxes[0].isCheckable()
    state = dict(recorded[2]["base selected"].state)
    idle_window.current_layout.on_state({**state, "movement": True})
    assert boxes[0].isChecked()
    idle_window.current_layout.on_state({**state, "movement": False})
    assert not boxes[0].isChecked()


def test_pointer_positions_go_to_the_worker_one_at_a_time(idle_window, recorded):
    win = idle_window
    frame = recorded[2]["base selected"]
    win._on_frame(frame, 0)
    win._on_hover((10.0, 20.0))
    assert win.worker.sent == 0  # movement is off
    win.state = {**frame.state, "movement": True, "ring_in": 3}
    win._on_hover((10.0, 20.0))
    assert win.worker.sent == 1 and win.worker.commands.get_nowait() == ("board.hover", {"x": 10.0, "y": 20.0})
    win._on_hover((11.0, 20.0))
    win._on_hover(None)  # only the newest waits
    assert win.worker.sent == 1
    win._on_frame(frame, 1)  # the worker did the first one
    assert win.worker.sent == 2 and win.worker.commands.get_nowait() == ("board.hover", {"x": None, "y": None})


def test_video_widget_reports_the_pointer(qapp):
    video = VideoWidget()
    video.resize(400, 300)
    video.set_image(np.zeros((300, 400, 3), np.uint8))
    seen = []
    video.hovered.connect(seen.append)
    assert video.hasMouseTracking()
    video.mouseMoveEvent(QMouseEvent(QEvent.Type.MouseMove, QPointF(100.0, 50.0), QPointF(100.0, 50.0), Qt.MouseButton.NoButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier))
    video.leaveEvent(None)
    assert seen == [(100.0, 50.0), None]


def test_layout_controls_never_take_the_focus(idle_window):
    root = idle_window.centralWidget()
    for control in root.findChildren(QAbstractButton) + root.findChildren(QAbstractItemView):
        assert control.focusPolicy() is Qt.FocusPolicy.NoFocus


def test_a_layout_error_keeps_the_frame_flow(idle_window, recorded):
    def boom(_state):
        raise RuntimeError("layout bug")

    idle_window.current_layout.on_state = boom
    idle_window.worker._gui_free.clear()
    with pytest.raises(RuntimeError):
        idle_window._on_frame(recorded[2]["live"], 0)
    assert idle_window.worker._gui_free.is_set()


# --- the panel layout ---------------------------------------------------------------


def test_panel_has_one_control_per_action_without_args_or_control_flag(idle_panel):
    assert set(idle_panel.current_layout.controls) == {a.id for a in ACTIONS if not a.args and a.control}


def test_panel_shows_debug_values(idle_panel, recorded):
    idle_panel._on_frame(recorded[2]["live"], 0)
    assert "Tags found: 4" in idle_panel.current_layout.debug.text()


def test_panel_controls_follow_the_state(idle_panel, recorded):
    for frame in recorded[2].values():
        idle_panel._on_frame(frame, 0)
        for action, control in idle_panel.current_layout.controls.values():
            assert control.isEnabled() is action.enabled(frame.state)
            if action.checked is not None:
                assert control.isChecked() is action.checked(frame.state)


def test_panel_button_runs_the_action(window):
    grid = window.state["grid"]
    window.current_layout.controls["view.grid"][1].click()
    wait_until(lambda: window.state["grid"] is (not grid))
    assert "snapshot.bases" not in window.current_layout.controls  # key and menu only
    window.shortcuts["snapshot.bases"].activated.emit()
    wait_until(lambda: window.state["mode"] == "snapshot" and not window.state["busy"])
    assert "snapshot.back" not in window.current_layout.controls  # Esc: key and menu only
    window.current_layout.controls["view.live"][1].click()
    wait_until(lambda: window.state["mode"] == "live")


def test_panel_quit_button_closes_the_window(qapp, tmp_path):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    win = MainWindow(with_layout(make_session(tmp_path, cfg, frame), "panel"), FakeCamera(frame))
    win.show()
    win.start()
    wait_until(lambda: win.state is not None)
    win.current_layout.controls["app.quit"][1].click()
    assert win.worker.isFinished() and win.camera.released


# --- layout choice in the window ----------------------------------------------------


class OtherLayout(PanelLayout):
    name = "other"


def queued_notices(win: MainWindow) -> list[str]:
    return [win.worker.notices.get_nowait() for _ in range(win.worker.notices.qsize())]


def test_unknown_layout_of_a_saved_state_gives_a_notice_and_the_default(qapp, tmp_path, recorded):
    session, frame, _ = recorded
    win = MainWindow(with_layout(session, "no-such"), FakeCamera(frame))
    assert win.current_layout.name == DEFAULT_LAYOUT
    assert "unknown layout" in " ".join(queued_notices(win))
    win.close()


def test_the_window_tells_its_layout_to_the_session(qapp, tmp_path, recorded):
    session, frame, _ = recorded
    win = MainWindow(session, FakeCamera(frame))
    assert session.ui_layout == DEFAULT_LAYOUT
    win.close()


def test_zoomed_window_gets_a_sharp_region_and_clicks_still_select(window):
    state = open_names(window)
    video = window.video
    video.zoom_by(3.0, widget_pos(window, state["bases"][1]["position_px"]))
    wait_until(lambda: window.worker.session.image_region is not None and video._region[2] < video._whole[0] * 0.5)
    assert video._image.width() > video._region[2]  # the region has more pixels than at 100%: drawn at the screen scale
    mouse_click(video, widget_pos(window, state["bases"][1]["position_px"]))
    wait_until(lambda: window.state["selected"] == 1)
    video.reset_zoom()
    wait_until(lambda: window.worker.session.image_region is None and video._region[2] == video._whole[0])


def test_layout_change_keeps_worker_snapshot_and_selection(window, monkeypatch):
    monkeypatch.setitem(LAYOUTS, "other", OtherLayout)
    state = open_names(window)
    x, y = state["bases"][1]["position_px"]
    window.dispatch("board.click", x=x, y=y)
    wait_until(lambda: window.state["selected"] == 1)
    old_video, thread = window.video, window.worker
    window.set_layout("other")

    assert window.current_layout.name == "other" and window.video is not old_video
    assert window.worker is thread and thread.isRunning() and not window.camera.released
    assert window.video._image is not None  # shows the last frame at once
    done = thread.done
    wait_until(lambda: thread.done == done and window.state["mode"] == "snapshot" and window.state["selected"] == 1 and window.current_layout.controls["names.accept"][1].isEnabled())
    wait_until(lambda: window.worker.session.ui_layout == "other")
    wait_until(lambda: thread.display_width == int(window.video.width() * window.video.devicePixelRatioF()))

    sent = thread.sent  # the new video is connected
    mouse_click(window.video, QPointF(window.video.width() / 2, window.video.height() / 2))
    assert thread.sent == sent + 1


def test_layout_change_closes_the_popup(window, monkeypatch):
    monkeypatch.setitem(LAYOUTS, "other", OtherLayout)
    state = open_names(window)
    mouse_click(window.video, widget_pos(window, state["bases"][0]["position_px"]), Qt.MouseButton.RightButton)
    popup = wait_until(lambda: window.popup)
    wait_until(lambda: window.state["selected"] == 0)
    window.set_layout("other")
    assert window.popup is None and not popup.isVisible()
    time.sleep(0.05)
    wait_until(lambda: window.worker.done == window.worker.sent)
    assert window.state["selected"] == 0


def test_l_key_goes_to_the_next_layout_and_tells_the_session(window):
    assert window.current_layout.name == "panel"
    press(window, "app.layout")
    assert window.current_layout.name == "rosters"
    wait_until(lambda: window.worker.session.ui_layout == "rosters")
    assert window.layout_items["rosters"].isChecked()
    press(window, "app.layout")
    assert isinstance(window.current_layout, InspectorLayout) and window.layout_items["inspector"].isChecked()
    press(window, "app.layout")
    assert window.current_layout.name == "panel" and window.layout_items["panel"].isChecked()


def test_view_menu_changes_the_layout(window):
    window.layout_items["inspector"].trigger()
    assert window.current_layout.name == "inspector"
    wait_until(lambda: window.status_label.text().startswith("4 corners OK"))


def test_the_opened_layout_comes_from_the_file(inspector):
    assert inspector.current_layout.name == "inspector" and inspector.layout_items["inspector"].isChecked()


def test_change_to_inspector_shows_the_selected_base(window):
    state = open_names(window)
    x, y = state["bases"][2]["position_px"]
    window.dispatch("board.click", x=x, y=y)
    wait_until(lambda: window.state["selected"] == 2)
    window.set_layout("inspector")
    layout = window.current_layout
    wait_until(lambda: layout.stack.currentWidget() is layout.pages["selected"] and layout.candidates.count() > 0)


# --- the inspector layout -----------------------------------------------------------


def control_of(win: MainWindow, action_id: str) -> QAbstractButton:
    """The control of an action on the page that shows."""
    return next(c for a, c in win.current_layout.controls if a.id == action_id and c.isVisibleTo(win))


def row_click(view: QAbstractItemView, row: int) -> None:
    view.itemClicked.emit(view.item(row))


@pytest.mark.parametrize(
    "name, page",
    [("live", "live"), ("busy", "live"), ("camera down", "live"), ("before the first frame", "live"), ("snapshot", "snapshot"), ("base selected", "selected"), ("ring", "selected"), ("target", "selected"), ("base selected, no names", "selected")],
)
def test_inspector_page_follows_the_state(idle_inspector, recorded, name, page):
    frame = recorded[2][name]
    idle_inspector._on_frame(frame, 0)
    layout = idle_inspector.current_layout
    assert page_for(frame.state) == page and layout.stack.currentWidget() is layout.pages[page]


def test_inspector_shows_an_opening(idle_inspector, recorded):
    layout = idle_inspector.current_layout
    live = recorded[2]["live"]
    state = dict(live.state, selected_opening={"index": 0, "kind": "breach", "open": True})
    assert page_for(state) == "opening"
    idle_inspector._on_frame(replace(live, state=state), 0)
    assert layout.stack.currentWidget() is layout.pages["opening"] and layout.opening_title.text() == "Breach point"
    assert layout.opening.state_label.text() == "Open" and layout.opening.button.text() == "Close"


def test_inspector_summary_text():
    bases = [{"label": "a"}, {"label": None}]
    assert summary_text(dict(snapshot_ok=False, bases=[], naming=False, unconfirmed=0)) == "Snapshot failed"
    assert summary_text(dict(snapshot_ok=True, bases=bases, naming=False, unconfirmed=0)) == "2 bases"
    assert summary_text(dict(snapshot_ok=True, bases=bases, naming=True, unconfirmed=1)) == "1/2 named, 1 not confirmed"


def test_inspector_controls_follow_the_state(idle_inspector, recorded):
    for frame in recorded[2].values():
        idle_inspector._on_frame(frame, 0)
        for action, control in idle_inspector.current_layout.controls:
            assert control.isEnabled() is action.enabled(frame.state)


def test_inspector_debug_opens_with_the_toggle(qapp, tmp_path, recorded):
    session, frame, _ = recorded
    win = MainWindow(with_layout(session, "inspector"), FakeCamera(frame))
    win.show()
    layout = win.current_layout
    win._on_frame(recorded[2]["live"], 0)
    assert not layout.debug.isVisible()
    layout.debug_toggle.click()
    assert layout.debug.isVisible() and "Tags found: 4" in layout.debug.text()
    layout.debug_toggle.click()
    assert not layout.debug.isVisible()
    win.close()


def test_inspector_list_does_not_fill_again_with_the_same_data(idle_inspector, recorded):
    idle_inspector._on_frame(recorded[2]["snapshot"], 0)
    view = idle_inspector.current_layout.operatives
    first = view.item(0)
    idle_inspector._on_frame(recorded[2]["snapshot"], 0)
    assert view.item(0) is first


def test_inspector_buttons_run_actions(inspector):
    assert all(action.id != "snapshot.bases" for action, _ in inspector.current_layout.controls)
    inspector.shortcuts["snapshot.bases"].activated.emit()
    wait_until(lambda: inspector.state["mode"] == "snapshot" and not inspector.state["busy"])
    layout = inspector.current_layout
    wait_until(lambda: layout.stack.currentWidget() is layout.pages["snapshot"] and layout.operatives.count() == 3)
    assert layout.summary.text() == "3 bases"
    control_of(inspector, "view.live").click()
    wait_until(lambda: inspector.state["mode"] == "live")
    control_of(inspector, "view.snapshot").click()
    wait_until(lambda: inspector.state["mode"] == "snapshot")


def test_inspector_shows_ring_buttons_and_the_distance(idle_inspector, recorded):
    layout = idle_inspector.current_layout
    idle_inspector._on_frame(recorded[2]["base selected"], 0)
    assert layout.distance.text() == "Right click a second base to measure to."
    idle_inspector._on_frame(recorded[2]["target"], 0)
    text = layout.distance.text()
    assert text.startswith("Rim to rim: ") and text.endswith('"') and "mm" not in text
    six = next(c for a, c in layout.controls if a.id == "ring.6")
    assert six.isChecked() and six.isEnabled() and six.text() == "6"


def test_inspector_ring_button_runs_the_action(inspector):
    state = open_names(inspector)
    mouse_click(inspector.video, widget_pos(inspector, state["bases"][0]["position_px"]))
    layout = inspector.current_layout
    wait_until(lambda: layout.stack.currentWidget() is layout.pages["selected"])
    wait_until(lambda: control_of(inspector, "ring.3").isEnabled())
    control_of(inspector, "ring.3").click()
    wait_until(lambda: inspector.state["ring_in"] == 3)


def test_inspector_right_click_shows_crop_and_list_and_no_popup(inspector):
    state = open_names(inspector)
    mouse_click(inspector.video, widget_pos(inspector, state["bases"][1]["position_px"]), Qt.MouseButton.RightButton)
    layout = inspector.current_layout
    wait_until(lambda: layout.stack.currentWidget() is layout.pages["selected"] and layout.candidates.count() > 0)
    wait_until(lambda: inspector._pending_click is None)
    assert inspector.popup is None and layout.crop._image is not None and "mm base" in layout.base_title.text()
    label = inspector.state["candidates"][0]["label"]
    assert layout.candidates.item(0).text().startswith(label)
    row_click(layout.candidates, 0)
    wait_until(lambda: inspector.state["bases"][1]["confirmed"] and inspector.state["selected"] == 1)  # the named base stays selected
    assert inspector.state["bases"][1]["label"] == label
    wait_until(lambda: layout.stack.currentWidget() is layout.pages["selected"])


def test_inspector_operative_row_selects_the_base_and_esc_goes_back(inspector):
    open_names(inspector)
    layout = inspector.current_layout
    wait_until(lambda: layout.operatives.count() == 3)
    row_click(layout.operatives, 2)
    wait_until(lambda: inspector.state["selected"] == 2)
    press(inspector, "snapshot.back")  # back to the live video, the base stays selected
    wait_until(lambda: inspector.state["mode"] == "live" and inspector.state["selected"] == 2)
    assert layout.stack.currentWidget() is layout.pages["selected"]
    press(inspector, "snapshot.back")  # in the live video: clears the selection
    wait_until(lambda: inspector.state["selected"] is None)


def test_dispatch_returns_none_while_busy(window):
    assert window.dispatch("view.grid") == window.worker.sent
    window.state = dict(window.state, busy=True)
    assert window.dispatch("view.grid") is None


# --- the command line ---------------------------------------------------------------


def run_main_with(monkeypatch, tmp_path, cfg, frame, drive, extra=()):
    """`main --image-source` with `run_window` replaced by `drive(session, camera)`."""
    path = tmp_path / "shot.png"
    cv2.imwrite(str(path), frame)
    monkeypatch.setattr(app.BoardConfig, "load", lambda p: cfg)
    monkeypatch.setattr(shell, "run_window", drive)
    return app.main(["--image-source", str(path), "--gallery-root", str(tmp_path / "gallery"), *extra])


def test_image_source_runs_the_live_app(monkeypatch, tmp_path, named):
    cfg, frame = named
    seen = {}

    def drive(session: LiveSession, camera):
        for _ in range(8):
            session.step(camera.read())
        session.dispatch("snapshot.name")
        session.step(camera.read())
        seen.update(session.state())
        return 0

    assert run_main_with(monkeypatch, tmp_path, cfg, frame, drive) == 0
    assert seen["mode"] == "snapshot" and seen["naming"] and len(seen["bases"]) == 3 and seen["drift"]["ok"]


def test_image_source_does_not_use_the_saved_reference(monkeypatch, tmp_path):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    saved = tmp_path / "saved_empty.png"
    cv2.imwrite(str(saved), frame)
    monkeypatch.setattr(app, "DEFAULT_REFERENCE", saved)
    before = saved.stat().st_mtime_ns
    seen = {}

    def drive(session: LiveSession, camera):
        seen["path"] = session.reference_path
        session.step(camera.read())
        session.dispatch("reference.take")
        seen.update(session.state())
        return 0

    assert run_main_with(monkeypatch, tmp_path, cfg, frame, drive) == 0
    assert seen["path"] is None and seen["reference_set"] and "not saved" in seen["notice"]["text"]
    assert saved.stat().st_mtime_ns == before


def test_image_source_with_a_reference_file(monkeypatch, tmp_path):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    ref = tmp_path / "empty.png"
    cv2.imwrite(str(ref), frame)
    seen = {}

    def drive(session: LiveSession, camera):
        seen.update(session.state())
        return 0

    assert run_main_with(monkeypatch, tmp_path, cfg, frame, drive, extra=["--reference", str(ref)]) == 0
    assert seen["reference_set"]


def test_run_window_opens_and_closes(monkeypatch, tmp_path, qapp):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    session = make_session(tmp_path, cfg, frame)
    camera = FakeCamera(frame)
    start = MainWindow.start

    def start_then_close(self):
        start(self)
        QTimer.singleShot(150, self.close)

    monkeypatch.setattr(MainWindow, "start", start_then_close)
    assert shell.run_window(session, camera) == 0
    assert camera.released


def test_map_picker_fills_from_state_and_sends_the_choice(qapp, tmp_path):
    from battlesight.ui.parts import MapPicker

    sent = []
    picker = MapPicker(lambda action_id, **args: sent.append((action_id, args)))
    state = {"terrain_types": ["Tomb World"], "terrain_type": "Tomb World", "map_layouts": ["Layout 6", "Layout 7"], "map_layout": None}
    picker.show_state(state)
    assert [picker.layout_box.itemText(i) for i in range(picker.layout_box.count())] == ["None", "Layout 6", "Layout 7"]
    assert picker.layout_box.currentText() == "None" and picker.terrain.currentText() == "Tomb World"
    assert sent == []  # filling sends nothing
    picker.layout_box.activated.emit(2)
    picker.layout_box.activated.emit(0)
    picker.terrain.activated.emit(0)
    assert sent == [("map.layout", {"name": "Layout 7"}), ("map.layout", {"name": None}), ("map.terrain_type", {"name": "Tomb World"})]


# --- the rosters layout -------------------------------------------------------------


@pytest.fixture
def rosters(qapp, tmp_path, named):
    cfg, frame = named
    win = started_window(tmp_path, cfg, frame, "rosters")
    yield win
    win.close()


@pytest.fixture
def idle_rosters(qapp, tmp_path, recorded):
    session, frame, _ = recorded
    win = MainWindow(with_layout(session, "rosters"), FakeCamera(frame))
    win.resize(1400, 900)
    win.show()
    yield win
    win.close()


def test_rosters_is_the_default_layout():
    assert DEFAULT_LAYOUT == "rosters" and list(LAYOUTS)[0] == "rosters"


def test_scores_are_percentages():
    from battlesight.ui.parts import candidate_text, roster_row

    assert candidate_text({"label": "orks/Nob", "score": 0.781}) == "orks/Nob   78%"
    row = {"name": "Nob", "base_index": None, "score": None, "confirmed": False}
    assert roster_row(row) == ("Nob", "?", "", "none")
    assert roster_row(dict(row, base_index=1, score=0.78)) == ("Nob", "78%", "", "warn")
    name, mark, _wounds, level = roster_row(dict(row, base_index=1, score=0.78, confirmed=True))
    assert name == "Nob" and mark not in ("", "?") and "%" not in mark and level == "ok"


def test_rosters_shows_mode_badge_team_list_and_commands(idle_rosters, recorded):
    layout = idle_rosters.current_layout
    frames = recorded[2]
    idle_rosters._on_frame(frames["live"], 0)
    assert idle_rosters.video.badge[0] == "Live"
    assert not layout.accept.isVisibleTo(idle_rosters) and layout.add_operative.isEnabled()  # also before space
    left, right = layout.rosters
    assert left.team.currentText() == "orks" and right.team.currentText() == "-"
    assert [left.operatives.item(i).text() for i in range(3)] == ["Nob", "Grot", "Burna"]  # team.toml order
    assert [left.operatives.mark(i) for i in range(3)] == ["?", "?", "?"]
    assert right.team.model().item(1).isEnabled() is False  # orks is on the left side
    assert left.team.model().item(1).isEnabled() is True
    assert right.operatives.count() == 0
    assert layout.stack.currentWidget() is layout.pages["empty"]

    idle_rosters._on_frame(frames["snapshot"], 0)
    assert idle_rosters.video.badge[0] == "Snapshot"
    assert layout.accept.isVisibleTo(idle_rosters) and layout.accept.text() == "Accept all identification"
    assert all(left.operatives.mark(i).endswith("%") for i in range(3))
    snap = frames["snapshot"].state
    accepted = replace(frames["snapshot"], state=dict(snap, unconfirmed=0))
    idle_rosters._on_frame(accepted, 0)
    assert not layout.accept.isVisibleTo(idle_rosters)
    assert idle_rosters.video.badge[1:] == (MODE_SNAPSHOT_COLOR, "pause")
    idle_rosters._on_frame(replace(frames["snapshot"], state=dict(snap, minimap=True)), 0)
    assert idle_rosters.video.badge == ("Minimap", MODE_MINIMAP_COLOR, "square")
    idle_rosters._on_frame(frames["live"], 0)
    assert idle_rosters.video.badge == ("Live", MODE_LIVE_COLOR, "dot")
    for sign in ("dot", "pause", "square"):  # each sign paints
        idle_rosters.video.set_badge("Mode", "#ffffff", sign)
        idle_rosters.video.grab()


def test_a_frame_from_before_a_check_box_command_does_not_put_the_box_back(idle_rosters, recorded):
    layout = idle_rosters.current_layout
    live = recorded[2]["live"]
    box = next(control for action, control in layout.controls if action.id == "view.grid")
    idle_rosters._on_frame(live, 0)
    assert box.isChecked()
    box.click()  # the box changes at once, the command waits in the worker queue
    number = idle_rosters._check_sent
    assert number == idle_rosters.worker.sent and not box.isChecked()
    idle_rosters._on_frame(live, number - 1)  # a frame from before the command
    assert not box.isChecked()
    idle_rosters._on_frame(replace(live, state=dict(live.state, grid=False)), number)
    assert not box.isChecked()
    idle_rosters._on_frame(live, number)  # the core has the last word
    assert box.isChecked()


def test_rosters_inspector_shows_the_operative(idle_rosters, recorded):
    layout = idle_rosters.current_layout
    frame = recorded[2]["ring"]
    idle_rosters._on_frame(frame, 0)
    state = frame.state
    base = state["bases"][state["selected"]]
    team, name = base["label"].split("/")
    assert layout.title.text() == f"{team} - {name} ({base['diameter_mm']:g}mm)"
    assert layout.stack.currentWidget() is layout.pages["operative"] and layout.crop._image is not None
    assert layout.ring.slider.value() == 6 and layout.ring.value.text() == '6"'
    assert not layout.candidates.isVisibleTo(idle_rosters) and layout.close.isVisibleTo(idle_rosters)
    layout.change.click()
    assert layout.candidates.isVisibleTo(idle_rosters) and not layout.distance.isVisibleTo(idle_rosters) and layout.game.isVisibleTo(idle_rosters) and layout.candidates.item(0).text().endswith("%")
    idle_rosters._on_frame(recorded[2]["target"], 0)  # same base: the list stays open
    assert layout.candidates.isVisibleTo(idle_rosters) and layout.distance.text().startswith("Rim to rim: ")
    assert layout.change.text() == "Cancel"
    layout.change.click()  # back to the operative
    assert not layout.candidates.isVisibleTo(idle_rosters) and layout.ring.isVisibleTo(idle_rosters)
    idle_rosters._on_frame(recorded[2]["base selected, no names"], 0)
    assert layout.title.text().startswith("Base (") and not layout.change.isEnabled()
    assert layout.change.text() == "Change operative" and layout.remove.text() == "Remove ID"
    assert layout.remove.isVisibleTo(idle_rosters) and layout.remove.isEnabled()  # also without names
    idle_rosters._on_frame(recorded[2]["live"], 0)
    assert not layout.remove.isVisibleTo(idle_rosters)


def test_rosters_shoot_at_shows_cancel_the_pointer_crop_and_the_result(idle_rosters, recorded):
    layout = idle_rosters.current_layout
    frames = recorded[2]
    selected = frames["base selected"]
    idle_rosters._on_frame(selected, 0)
    order = [layout.title.parentWidget().layout().itemAt(0).layout().itemAt(k).widget() for k in range(6)]
    assert order[3:5] == [layout.remove, layout.shoot]  # after "Change operative", "Change ID"
    assert layout.shoot.text() == "Shoot at" and layout.shoot.isEnabled() and layout.shoot.isVisibleTo(idle_rosters)
    assert layout.ring.isVisibleTo(idle_rosters) and not layout.shot.isVisibleTo(idle_rosters)
    layout.change.click()  # the name list closes when the targeting lines start
    aim = np.full((80, 80, 3), 128, np.uint8)
    shooting = dict(selected.state, shooting=True, shot=None)
    idle_rosters._on_frame(replace(selected, state=shooting, crop=aim), 0)
    assert layout.shoot.text() == "Cancel" and not layout.change.isChecked()
    assert layout.shot.isVisibleTo(idle_rosters) and not layout.ring.isVisibleTo(idle_rosters) and layout.crop._image is not None
    assert not layout.obscured.isVisibleTo(idle_rosters) and not layout.in_cover.isVisibleTo(idle_rosters) and "Esc" in layout.shot_text.text()
    for obscured, in_cover in ((True, False), (False, True), (True, True)):
        state = dict(shooting, shot={"obscured": obscured, "in_cover": in_cover, "intervening": 1, "shooter": 1})
        idle_rosters._on_frame(replace(selected, state=state, crop=aim), 0)
        assert layout.obscured.isVisibleTo(idle_rosters) is obscured and layout.in_cover.isVisibleTo(idle_rosters) is in_cover
        assert layout.obscured.text() == "OBSCURED" and layout.in_cover.text() == "IN COVER"
    idle_rosters._on_frame(replace(selected, state=dict(shooting, shot={"obscured": False, "in_cover": False, "intervening": 0, "shooter": None}), crop=None), 0)
    assert layout.crop.isVisibleTo(idle_rosters)  # the pointer is off the board: the crop keeps its place
    sent = []
    idle_rosters.dispatch = lambda action_id, **args: sent.append(action_id)
    layout.shoot.click()
    assert sent == ["measure.shoot"]
    idle_rosters._on_frame(selected, 0)
    assert layout.shoot.text() == "Shoot at" and layout.ring.isVisibleTo(idle_rosters)


def test_locate_terrain_shows_the_pointer_crop_like_locate_operative(idle_rosters, idle_inspector, recorded):
    snap = recorded[2]["snapshot"]
    aim = np.full((80, 80, 3), 128, np.uint8)
    for window, title in ((idle_rosters, lambda layout: layout.title), (idle_inspector, lambda layout: layout.aim_title)):
        layout = window.current_layout
        window._on_frame(replace(snap, state=dict(snap.state, locating_light=True), crop=aim), 0)
        assert layout.stack.currentWidget() is layout.pages["adding"] and layout.aim_crop._array is aim
        assert title(layout).text() == "Locate terrain" and "light terrain piece" in layout.aim_hint.text()
        window._on_frame(replace(snap, state=dict(snap.state, adding=True), crop=aim), 0)
        assert title(layout).text() == "Locate operative" and "base of the operative" in layout.aim_hint.text()


def test_rosters_add_operative_shows_the_instruction_and_then_the_name_list(idle_rosters, recorded):
    layout = idle_rosters.current_layout
    frames = recorded[2]
    snap = frames["snapshot"]
    idle_rosters._on_frame(snap, 0)
    assert layout.add_operative.isEnabled() and not layout.add_operative.isCheckable()
    aim = np.full((80, 80, 3), 128, np.uint8)
    idle_rosters._on_frame(replace(snap, state=dict(snap.state, adding=True), crop=aim), 0)
    assert not layout.add_operative.isEnabled() and layout.stack.currentWidget() is layout.pages["adding"]
    assert layout.title.text() == "Locate operative" and layout.aim_crop._image is not None
    # The found base is selected and its name list is open at once.
    state = dict(frames["base selected"].state, new_base=frames["base selected"].state["selected"])
    idle_rosters._on_frame(replace(frames["base selected"], state=state), 0)
    assert layout.stack.currentWidget() is layout.pages["operative"] and layout.change.isChecked()
    assert layout.candidates.isVisibleTo(idle_rosters) and not layout.details.isVisibleTo(idle_rosters) and not layout.wounds.isVisibleTo(idle_rosters) and layout.game.isVisibleTo(idle_rosters)
    assert layout.change.text() == "Cancel"
    layout.change.click()  # Cancel: the name list closes, wounds, ring and distance come back
    assert not layout.candidates.isVisibleTo(idle_rosters) and layout.details.isVisibleTo(idle_rosters) and layout.wounds.isVisibleTo(idle_rosters)
    assert layout.change.text() == "Change operative"
    # A name choice closes the list; the operative stays in the inspector.
    idle_rosters._on_frame(replace(frames["base selected"], state=state), 0)
    layout.change.setChecked(True)
    layout.candidates.itemClicked.emit(layout.candidates.item(0))
    assert not layout.change.isChecked() and layout.details.isVisibleTo(idle_rosters)
    assert layout.stack.currentWidget() is layout.pages["operative"]


def test_rosters_inspector_shows_an_opening(idle_rosters, recorded):
    layout = idle_rosters.current_layout
    live = recorded[2]["live"]
    for kind, title, is_open, text in (("hatchway", "Hatchway", False, "Closed"), ("breach", "Breach point", True, "Open")):
        idle_rosters._on_frame(replace(live, state=dict(live.state, selected_opening={"index": 0, "kind": kind, "open": is_open})), 0)
        assert layout.stack.currentWidget() is layout.pages["opening"] and layout.title.text() == title
        assert layout.opening.button.isEnabled() and layout.opening.state_label.text() == text


LIGHT_PIECE = {"index": 0, "key": "C2", "center_mm": [100.0, 200.0], "angle_deg": 12.0, "located": True, "outline": True}


def test_inspectors_show_a_light_terrain_piece(idle_rosters, idle_inspector, recorded):
    live = recorded[2]["live"]
    image = np.zeros((60, 80, 3), np.uint8)  # the piece cut from the board (`LiveFrame.crop`)
    for window, title in ((idle_rosters, lambda layout: layout.title), (idle_inspector, lambda layout: layout.light_title)):
        layout = window.current_layout
        state = dict(live.state, light=[LIGHT_PIECE], selected_light=LIGHT_PIECE, light_can_locate=True, light_files=["C2"])
        window._on_frame(replace(live, state=state, crop=image), 0)
        control = layout.light
        assert layout.stack.currentWidget() is layout.pages["light"] and title(layout).text() == "Light terrain C2"
        assert control.pose_label.isHidden() and control.remove.isEnabled() and control.remove.text() == "Remove ID"
        heading = title(layout)
        centre_y = lambda w: w.mapTo(window, w.rect().center()).y()
        assert control.remove.parent() is not control and abs(centre_y(control.remove) - centre_y(heading)) < control.remove.height()  # in the title line
        assert control.remove.mapTo(window, control.remove.rect().topLeft()).x() > heading.mapTo(window, heading.rect().topRight()).x()  # after the title
        assert not control.crop.isHidden() and control.crop._array is image
        if window is idle_rosters:  # the low top bar: the image at the left, no overlap
            assert control.crop.geometry().right() < control.width() and control.crop.height() <= layout.stack.height()
        assert not control.remove.isHidden() and control.identify.isHidden()  # no disabled button
        buttons = control.findChildren(QAbstractButton) + [control.remove]
        assert set(buttons) == {control.remove, control.identify} and all(b.focusPolicy() == Qt.FocusPolicy.NoFocus for b in buttons)
        card = dict(LIGHT_PIECE, located=False)
        window._on_frame(replace(live, state=dict(state, light=[card], selected_light=card), crop=None), 0)
        assert control.pose_label.text().startswith("Not identified") and control.crop.isHidden()
        assert control.remove.isHidden() and not control.identify.isHidden() and control.identify.isEnabled() and control.identify.text() == "Identify terrain here"
        none = dict(LIGHT_PIECE, located=False, outline=False)  # no piece file of its key: the other files are searched
        window._on_frame(replace(live, state=dict(state, light=[none], selected_light=none, light_can_locate=False)), 0)
        assert "No outline file" in control.pose_label.text() and control.remove.isHidden() and control.identify.isEnabled()
        window._on_frame(replace(live, state=dict(state, light=[none], selected_light=none, light_files=[])), 0)
        assert not control.identify.isEnabled()  # no piece file at all
        window._on_frame(replace(live, state=state), 0)
        window._on_frame(replace(live, state=dict(live.state, selected_light=None)), 0)
        assert not control.remove.isVisible()  # not in the title line of another page


def test_light_control_sends_its_actions(qapp):
    from battlesight.ui.parts import LightControl

    sent = []
    control = LightControl(lambda action_id, **args: sent.append((action_id, args)))
    control.remove.click()
    control.identify.click()
    assert sent == [("light.remove", {}), ("light.locate_here", {})]


def test_rosters_row_slider_and_name_fix_run_actions(rosters):
    open_names(rosters)
    layout = rosters.current_layout
    left = layout.rosters[0]
    wait_until(lambda: left.operatives.count() == 3 and left.operatives.mark(0).endswith("%"))
    rows = rosters.state["rosters"][0]["operatives"]
    row_click(left.operatives, 1)
    wait_until(lambda: rosters.state["selected"] == rows[1]["base_index"])
    wait_until(lambda: layout.stack.currentWidget() is layout.pages["operative"] and layout.ring.isEnabled())
    layout.ring.slider.setValue(4)
    wait_until(lambda: rosters.state["ring_in"] == 4)
    layout.ring.slider.setValue(0)
    wait_until(lambda: rosters.state["ring_in"] is None)
    layout.change.click()
    wait_until(lambda: layout.candidates.count() > 0)
    index = rosters.state["selected"]
    label = rosters.state["candidates"][0]["label"]
    row_click(layout.candidates, 0)
    wait_until(lambda: rosters.state["bases"][index]["confirmed"] and rosters.state["bases"][index]["label"] == label)
    press(rosters, "snapshot.back")
    wait_until(lambda: rosters.state["mode"] == "live" and rosters.video.badge[0] == "Live")
    assert any(b["label"] for b in rosters.state["bases"])  # the names stay on the live video
    layout.close.click()
    wait_until(lambda: rosters.state["selected"] is None and layout.stack.currentWidget() is layout.pages["empty"])


def test_rosters_remove_id_drops_the_base(rosters):
    open_names(rosters)
    layout = rosters.current_layout
    count = len(rosters.state["bases"])
    rosters.dispatch("base.select", index=0)
    wait_until(lambda: rosters.state["selected"] == 0 and layout.remove.isVisibleTo(rosters))
    layout.remove.click()
    wait_until(lambda: len(rosters.state["bases"]) == count - 1 and rosters.state["selected"] is None)
    wait_until(lambda: layout.stack.currentWidget() is layout.pages["empty"])


def test_rosters_team_choice_sends_the_action(rosters):
    right = rosters.current_layout.rosters[1]
    assert right.team.itemText(1) == "orks"
    right.team.activated.emit(1)  # orks is on the left already
    wait_until(lambda: rosters.state["notice"] is not None and "other side" in rosters.state["notice"]["text"])
    rosters.current_layout.rosters[0].team.activated.emit(0)
    wait_until(lambda: rosters.state["team_slots"] == [None, None])
    right.team.activated.emit(1)
    wait_until(lambda: rosters.state["team_slots"] == [None, "orks"] and right.operatives.count() == 3)


def test_rosters_debug_goes_to_the_status_bar(idle_rosters, recorded):
    layout = idle_rosters.current_layout
    idle_rosters._on_frame(recorded[2]["live"], 0)
    assert not idle_rosters.debug_label.isVisible()
    assert layout.debug_toggle.text() == "Debug" and not layout.debug_toggle.isChecked()
    assert layout.debug_toggle.parent().title() == "Commands"  # the last control of Commands, no Debug section
    assert "Debug" not in [box.title() for box in idle_rosters.findChildren(QGroupBox)]
    layout.debug_toggle.click()
    assert idle_rosters.debug_label.isVisible() and idle_rosters.debug_label.text().startswith("fit ")
    assert "tags" not in idle_rosters.debug_label.text() and layout.debug_toggle.isChecked()
    idle_rosters.set_layout("panel")
    assert not idle_rosters.debug_label.isVisible()


# --- game state: wounds and openings ----------------------------------------------


def test_roster_row_shows_the_wounds():
    from battlesight.ui.parts import DEAD, roster_row, wounds_text

    row = {"name": "Nob", "base_index": None, "score": None, "confirmed": False, "wounds": 7, "max_wounds": 12}
    assert wounds_text(7, 12) == "7/12" and wounds_text(None, None) == ""
    assert roster_row(row) == ("Nob", "?", "7/12", "none")
    assert roster_row(dict(row, base_index=1, score=0.78)) == ("Nob", "78%", "7/12", "warn")
    assert roster_row(dict(row, wounds=None, max_wounds=None)) == ("Nob", "?", "", "none")
    assert roster_row(dict(row, wounds=0, base_index=1, confirmed=True, dead=True)) == ("Nob", DEAD, "0/12", "ok")  # skull for a dead operative


@pytest.fixture
def wounded_rosters(qapp, tmp_path, named):
    from battlesight import ident as idt

    cfg, frame = named
    team = tmp_path / "gallery" / "orks"
    idt.save_team_file(team, "orks", idt.load_team_file(team), {"Nob": 5, "Burna": 12})
    win = started_window(tmp_path, cfg, frame, "rosters")
    yield win
    win.close()


def test_rosters_wounds_control_and_team_list(wounded_rosters):
    win = wounded_rosters
    open_names(win)
    layout = win.current_layout
    left = layout.rosters[0]
    rows = win.state["rosters"][0]["operatives"]
    nob = next(k for k, r in enumerate(rows) if r["name"] == "Nob")
    wait_until(lambda: left.operatives.wounds(nob) == "5/5")
    row_click(left.operatives, nob)
    wounds = layout.wounds
    wait_until(lambda: wounds.isVisibleTo(win) and wounds.controls.isVisibleTo(win) and wounds.value.value() == 5)
    assert not wounds.plus.isEnabled() and wounds.minus.isEnabled() and wounds.maximum.text() == "/ 5"
    wounds.minus.click()
    wait_until(lambda: win.state["selected_wounds"]["current"] == 4 and wounds.value.value() == 4)
    wounds.value.setValue(2)
    wounds.value.editingFinished.emit()
    wait_until(lambda: win.state["selected_wounds"]["current"] == 2 and left.operatives.wounds(nob) == "2/5")
    press(win, "wounds.plus")
    wait_until(lambda: win.state["selected_wounds"]["current"] == 3)
    grot = next(k for k, r in enumerate(rows) if r["name"] == "Grot")
    row_click(left.operatives, grot)
    wait_until(lambda: wounds.warning.isVisibleTo(win) and not wounds.controls.isVisibleTo(win))
    assert "no maximum wounds" in wounds.warning.text()


def test_rosters_opening_button_opens_and_closes(qapp, tmp_path, named):
    from .test_live import opening_px, write_map

    cfg, frame = named
    write_map(tmp_path / "maps")
    win = started_window(tmp_path, cfg, frame, "rosters")
    try:
        win.dispatch("map.terrain_type", name="Test World")
        win.dispatch("map.layout", name="Small")
        wait_until(lambda: win.state["map_layout"] == "Small")
        x, y = opening_px(win.worker.session)
        win.dispatch("board.click", x=x, y=y)
        layout = win.current_layout
        wait_until(lambda: layout.stack.currentWidget() is layout.pages["opening"])
        control = layout.opening
        assert control.state_label.text() == "Closed" and control.button.text() == "Open" and control.button.isEnabled()
        control.button.click()
        wait_until(lambda: win.state["selected_opening"]["open"] and control.state_label.text() == "Open")
        assert control.button.text() == "Close"
        press(win, "opening.toggle")
        wait_until(lambda: not win.state["openings"][0]["open"])
    finally:
        win.close()


# --- score counter ------------------------------------------------------------------


def test_team_list_mark_column_fits_the_widest_mark(qapp):
    from PySide6.QtGui import QFontMetrics

    from battlesight.ui.parts import MARK_PAD, _RosterList

    rows = _RosterList(lambda *a, **k: None)
    metrics = QFontMetrics(rows.font())
    rows.fill_rows([{}, {}], [("Nob", "?", "", "none"), ("Grot", "?", "", "none")])
    narrow = rows.delegate.mark_width(metrics)
    assert narrow == metrics.horizontalAdvance("?") + 2 * MARK_PAD
    rows.fill_rows([{}, {"x": 1}], [("Nob", "?", "", "none"), ("Grot", "78%", "", "warn")])
    assert rows.delegate.mark_width(metrics) == metrics.horizontalAdvance("78%") + 2 * MARK_PAD > narrow
    rows.fill_rows([], [])
    assert rows.delegate.mark_width(metrics) == MARK_PAD


def test_score_grid_lines_are_drawn_once():
    from battlesight.ui.parts import cell_border

    assert cell_border("kill", None) == ""
    assert all(s in cell_border("crit", 2) for s in ("left", "top", "right", "bottom"))
    tac3 = cell_border("tac", 3)
    assert "left" not in tac3 and "top" not in tac3 and "right" in tac3 and "bottom" in tac3


def test_total_text_and_team_color():
    from battlesight.ui.parts import team_color, total_text

    assert total_text({"total": 15, "primary_bonus": 3}) == "15 VP"
    assert team_color(0) == qt_hex(TEAM_1_COLOR) and team_color(1) == qt_hex(TEAM_2_COLOR)  # the overlay colours of the teams


def test_rosters_score_counter(qapp, tmp_path, named):
    from battlesight import ident as idt
    from battlesight import score as sc

    cfg, frame = named
    team = tmp_path / "gallery" / "orks"
    idt.save_team_file(team, "orks", idt.load_team_file(team), None, idt.TeamInfo(("Recon",)))
    win = started_window(tmp_path, cfg, frame, "rosters")
    try:
        left, right = win.current_layout.counters
        assert left.isVisibleTo(win) and right.isVisibleTo(win)
        wait_until(lambda: left.total.text() == "0 VP" and left.cp.text() == "3")
        assert [left.tac_op.itemText(i) for i in range(1, left.tac_op.count())] == list(sc.TAC_OPS["Recon"])
        assert not right.tac_op.isEnabled()  # no team on the right
        left.skulls[("crit", 3)][1].click()
        wait_until(lambda: win.state["scores"][0]["cells"]["crit"] == [0, 2, 0] and left.lit("crit", 3) == 2)
        left.skulls[("crit", 3)][1].click()  # the last lit skull turns off
        wait_until(lambda: left.lit("crit", 3) == 1 and left.points["crit"].text() == "1")
        assert len(left.skulls[("kill", None)]) == sc.KILL_GRADE_MAX + 1  # grades and the bonus skull
        left.skulls[("tac", 2)][1].click()
        wait_until(lambda: left.total.text() == "3 VP")
        left.primary.activated.emit(sc.PRIMARIES.index("tac"))
        wait_until(lambda: left.total.text() == "4 VP" and left.bonus.text() == "+1")
        left.show_state({"scores": [dict(win.state["scores"][0], kill_grade=2, kill_bonus=1), win.state["scores"][1]]})
        assert left.lit("kill", None) == 3 and left.skulls[("kill", None)][-1].property("lit")  # Kill Op is automatic
        assert right.bonus.text() == "+0" and qt_hex(TEAM_1_COLOR) in left.total.styleSheet() and not left.title()
        left.tac_op.activated.emit(2)
        wait_until(lambda: win.state["scores"][0]["tac_op"] == sc.TAC_OPS["Recon"][1])
        left.cp_minus.click()
        right.cp_plus.click()
        wait_until(lambda: left.cp.text() == "2" and right.cp.text() == "4")
        assert right.total.text() == "0 VP"
    finally:
        win.close()


def test_rosters_game_bar(qapp, tmp_path, named):
    from battlesight import score as sc

    cfg, frame = named
    win = started_window(tmp_path, cfg, frame, "rosters")
    try:
        layout = win.current_layout
        game = layout.game
        left, right = (roster.team for roster in layout.rosters)
        wait_until(lambda: game.turn.text() == "1")
        assert left.isVisibleTo(win) and right.isVisibleTo(win) and game.isVisibleTo(win)
        assert left.parentWidget() is right.parentWidget() is game.parentWidget()  # one bar
        assert left.geometry().right() < game.geometry().left() and game.geometry().right() < right.geometry().left()
        assert not game.turn_minus.isEnabled()
        for _ in range(3):
            game.turn_plus.click()
        wait_until(lambda: game.turn.text() == "4" and not game.turn_plus.isEnabled())
        game.turn_minus.click()
        wait_until(lambda: win.state["turn"] == 3 and game.turn.text() == "3")
        assert [game.crit_op.itemText(i) for i in range(1, game.crit_op.count())] == list(sc.CRIT_OPS)
        game.crit_op.activated.emit(2)
        wait_until(lambda: win.state["crit_op"] == sc.CRIT_OPS[1] and game.crit_op.currentIndex() == 2)
        game.crit_op.activated.emit(0)
        wait_until(lambda: win.state["crit_op"] is None)
    finally:
        win.close()


def test_rosters_select_an_operative_from_the_team_list(qapp, tmp_path, named):
    from battlesight import ident as idt
    from battlesight.ui.parts import COLORS, team_color

    cfg, frame = named
    team = tmp_path / "gallery" / "orks"
    idt.save_team_file(team, "orks", idt.load_team_file(team), {"Nob": 5})
    win = started_window(tmp_path, cfg, frame, "rosters")
    try:
        layout = win.current_layout
        left = layout.rosters[0].operatives
        wait_until(lambda: left.count() == 3)
        nob = next(k for k, r in enumerate(win.state["rosters"][0]["operatives"]) if r["name"] == "Nob")
        assert left.item(nob).foreground().color().name() == COLORS["none"]  # not identified: grey
        row_click(left, nob)  # no base: name and wounds only
        wait_until(lambda: layout.stack.currentWidget() is layout.pages["roster_operative"])
        assert layout.title.text().startswith("orks - Nob (") and layout.title.text().endswith("mm)")
        assert layout.close.isVisibleTo(win) and not layout.change.isVisibleTo(win) and not layout.remove.isVisibleTo(win)
        wait_until(lambda: left.is_selected(nob) and layout.roster_wounds.value.value() == 5)
        assert sum(left.is_selected(k) for k in range(3)) == 1
        layout.roster_wounds.minus.click()
        wait_until(lambda: left.wounds(nob) == "4/5")
        layout.close.click()
        wait_until(lambda: layout.stack.currentWidget() is layout.pages["empty"] and not left.is_selected(nob))

        open_names(win)
        wait_until(lambda: left.mark(nob).endswith("%"))
        assert left.item(nob).foreground().color().name() == team_color(0)  # identified: team colour
        row_click(left, nob)
        wait_until(lambda: layout.stack.currentWidget() is layout.pages["operative"] and left.is_selected(nob))
    finally:
        win.close()


def test_rosters_add_and_remove_operative_buttons(rosters):
    win = rosters
    left = win.current_layout.rosters[0]
    wait_until(lambda: left.operatives.count() == 3)
    assert not left.add.isEnabled() and not left.remove.isEnabled() and not left.chooser.isVisibleTo(win)
    row_click(left.operatives, 0)  # Nob
    wait_until(lambda: left.remove.isEnabled())
    left.remove.click()
    wait_until(lambda: left.operatives.count() == 2 and left.add.isEnabled())
    assert not left.remove.isEnabled()  # the selection went with the operative
    left.add.click()
    assert left.chooser.isVisibleTo(win) and left.add.text() == "Cancel" and left.chooser.item(0).text() == "Nob"
    left.add.click()  # Cancel
    assert not left.chooser.isVisibleTo(win) and left.add.text() == "Add operative"
    left.add.click()
    left.chooser.itemClicked.emit(left.chooser.item(0))
    assert not left.chooser.isVisibleTo(win) and left.add.text() == "Add operative"
    wait_until(lambda: left.operatives.count() == 3 and left.operatives.item(2).text() == "Nob" and not left.add.isEnabled())


def test_rosters_commands_has_locate_operative(rosters):
    assert rosters.current_layout.add_operative.text() == "Locate operative"


def test_texts_of_copies():
    from battlesight.ui.parts import candidate_text, copy_text, missing_text

    assert missing_text({"label": "orks/Nob", "name": "Nob", "max": 1, "copies_in_play": 0}) == "Nob"
    assert missing_text({"label": "orks/Grot", "name": "Grot", "max": 3, "copies_in_play": 2}) == "Grot (3/3)"
    assert candidate_text({"label": "orks/Nob", "name": "Nob", "score": None}) == "orks/Nob"
    assert copy_text({"label": "orks/Grot#2", "name": "Grot #2", "copy": 2, "wounds": 3, "max_wounds": 5}) == "Grot #2 (3/5)"


def test_texts_of_models():
    from battlesight.ui.parts import candidate_text, model_text

    assert candidate_text({"label": "necrons/Guardian - tesla", "name": "Guardian - tesla", "score": 0.81, "locked": True}) == "necrons/Guardian - tesla   81%   (locked)"
    assert candidate_text({"label": "necrons/Guardian", "name": "Auto model", "score": None, "auto": True, "current": True}) == "Auto model  ✓"
    base = {"model_name": "Guardian - tesla", "model_locked": False}
    assert model_text({"selected": 0, "bases": [base]}) == "Model: Guardian - tesla"
    assert model_text({"selected": 0, "bases": [{"model_name": None, "model_locked": False}]}) == ""


def test_status_bar_progress_of_the_model_load(idle_rosters, recorded):
    """The progress bar is at the right end of the status bar, after the
    debug values, with one text for the task. The commands that name
    operatives are off until the end."""
    layout, progress = idle_rosters.current_layout, idle_rosters.progress
    idle_rosters._on_frame(recorded[2]["loading model"], 0)
    assert not progress.isHidden()
    assert (progress.label.text(), progress.bar.value(), progress.bar.maximum()) == ("Loading models:", 2, 5)
    idle_rosters.show_debug(True)
    QApplication.processEvents()
    assert progress.parent() is idle_rosters.statusBar() and progress.x() > idle_rosters.debug_label.x()  # right of the debug values
    assert not layout.add_operative.isEnabled()
    assert not idle_rosters.menu_actions["snapshot.name"][1].isEnabled()
    idle_rosters._on_frame(recorded[2]["live"], 0)
    assert progress.isHidden()
    assert layout.add_operative.isEnabled()
    assert idle_rosters.menu_actions["snapshot.name"][1].isEnabled()


def test_progress_of_a_long_action_reaches_the_status_bar(rosters):
    """"Identify all operatives": the worker sends each step; the bar is hidden at the end."""
    seen = []
    rosters.worker.progress_ready.connect(lambda p: seen.append((p["text"], p["done"], p["total"])))
    open_names(rosters)
    wait_until(lambda: seen and seen[-1][1] == seen[-1][2])
    assert {text for text, _, _ in seen} == {"Identifying operatives"}
    wait_until(lambda: not rosters.state["busy"] and rosters.progress.isHidden())
