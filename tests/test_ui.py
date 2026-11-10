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
from PySide6.QtCore import QEvent, QPointF, Qt, QTimer  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QAbstractButton, QAbstractItemView, QApplication  # noqa: E402

import battlesight.app as app  # noqa: E402
from battlesight.live import ACTIONS, LiveSession  # noqa: E402
from battlesight.ui import shell  # noqa: E402
from battlesight.ui.layouts import DEFAULT_LAYOUT, LAYOUTS  # noqa: E402
from battlesight.ui.layouts.inspector import InspectorLayout, page_for, summary_text  # noqa: E402
from battlesight.ui.layouts.panel import PanelLayout  # noqa: E402
from battlesight.ui.parts import debug_text  # noqa: E402
from battlesight.ui.parts import VideoWidget  # noqa: E402
from battlesight.ui.settings import load_ui_settings, save_layout  # noqa: E402
from battlesight.ui.shell import MainWindow, help_text, status_text  # noqa: E402

from .test_ident import enroll_from, ident_cfg, make_gallery, paint_board  # noqa: E402
from .test_live import GAME, TRAIN, make_session, settled  # noqa: E402


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
def named_setup(tmp_path):
    cfg = ident_cfg()
    gallery = make_gallery(tmp_path, cfg)
    enroll_from(gallery, cfg, paint_board(cfg, TRAIN), {n: c for n, (c, _) in TRAIN.items()})
    return cfg, paint_board(cfg, GAME)


@pytest.fixture
def window(qapp, tmp_path, named_setup):
    cfg, frame = named_setup
    win = started_window(tmp_path, cfg, frame, "panel")
    yield win
    win.close()


@pytest.fixture
def inspector(qapp, tmp_path, named_setup):
    cfg, frame = named_setup
    win = started_window(tmp_path, cfg, frame, "inspector")
    yield win
    win.close()


def layout_file(tmp_path, name: str) -> Path:
    path = tmp_path / "ui.toml"
    path.write_text(f'[ui]\nlayout = "{name}"\n')
    return path


def started_window(tmp_path, cfg, frame, layout: str) -> MainWindow:
    win = MainWindow(make_session(tmp_path, cfg, frame, teams=("orks",)), FakeCamera(frame), layout_file(tmp_path, layout))
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
    state = dict(camera=True, drift=drift, lens="nominal", busy=False, mode="live", reference_set=True, snapshot_ok=None, naming=False, bases=[], ring_in=None, distance=None)
    state.update(over)
    return state


@pytest.mark.parametrize(
    "over, text, level",
    [
        (dict(camera=False), "camera down", "error"),
        (dict(drift=None), "waiting for the first frame", "warn"),
        (dict(), "4 corners OK  |  nominal lens", "ok"),
        (dict(busy=True), "working", "ok"),
        (dict(mode="snapshot", snapshot_ok=False), "snapshot failed", "ok"),
        (dict(mode="snapshot", snapshot_ok=True, naming=True, bases=[{"label": "a"}, {"label": None}]), "1/2 named", "ok"),
        (dict(mode="snapshot", snapshot_ok=True, bases=[{"label": None}]), "1 bases", "ok"),
        (dict(mode="snapshot", snapshot_ok=True, bases=[{"label": None}], ring_in=6), "ring 6 in", "ok"),
        (dict(mode="snapshot", snapshot_ok=True, bases=[{"label": None}], distance={"mm": 127.0, "in": 5.0}), 'rim to rim 5.00"', "ok"),
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
    action = popup.actions()[0]
    label = window.state["candidates"][0]["label"]
    assert action.text().startswith(label)
    action.trigger()
    popup.close()
    wait_until(lambda: window.state["bases"][1]["confirmed"])
    assert window.state["bases"][1]["label"] == label and window.state["selected"] is None


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
def test_ring_key_and_second_base_show_in_the_status_area_of_each_layout(qapp, tmp_path, named_setup, layout):
    cfg, frame = named_setup
    win = started_window(tmp_path, cfg, frame, layout)
    try:
        state = open_names(win)
        mouse_click(win.video, widget_pos(win, state["bases"][0]["position_px"]))
        wait_until(lambda: win.state["selected"] == 0)
        press(win, "ring.6")
        wait_until(lambda: win.state["ring_in"] == 6 and "ring 6 in" in win.status_label.text())
        mouse_click(win.video, widget_pos(win, state["bases"][1]["position_px"]))
        wait_until(lambda: win.state["target"] == 1 and "rim to rim" in win.status_label.text())
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


# --- data/ui.toml -------------------------------------------------------------------


def test_ui_settings_missing_file_is_not_a_problem(tmp_path):
    settings = load_ui_settings(tmp_path / "ui.toml")
    assert settings.layout is None and settings.problem is None and settings.options_for("panel") == {}


def test_ui_settings_reads_layout_and_options(tmp_path):
    path = tmp_path / "ui.toml"
    path.write_text('[ui]\nlayout = "panel"\n\n[layouts.panel]\npanel_side = "left"\n')
    settings = load_ui_settings(path)
    assert settings.layout == "panel" and settings.options_for("panel") == {"panel_side": "left"} and settings.problem is None


@pytest.mark.parametrize("text", ["not toml [", "ui = 3", "[ui]\nlayout = 3", "layouts = 3", "[layouts]\npanel = 3"])
def test_ui_settings_bad_file_gives_a_problem_and_defaults(tmp_path, text):
    path = tmp_path / "ui.toml"
    path.write_text(text)
    settings = load_ui_settings(path)
    assert settings.problem and settings.layout is None and settings.options_for("panel") == {}


def test_save_layout_makes_the_file(tmp_path):
    path = tmp_path / "sub" / "ui.toml"
    save_layout(path, "panel")
    assert load_ui_settings(path).layout == "panel"


@pytest.mark.parametrize(
    "before, after",
    [
        ('# mine\n[ui]\nlayout = "old"  # c\nother = 1\n\n[layouts.old]\nlayout = "keep"\n', '# mine\n[ui]\nlayout = "new"\nother = 1\n\n[layouts.old]\nlayout = "keep"\n'),
        ('[ui]\nother = 1\n', '[ui]\nlayout = "new"\nother = 1\n'),
        ('[layouts.old]\na = 1\n', '[ui]\nlayout = "new"\n\n[layouts.old]\na = 1\n'),
    ],
)
def test_save_layout_keeps_the_other_text(tmp_path, before, after):
    path = tmp_path / "ui.toml"
    path.write_text(before)
    save_layout(path, "new")
    assert path.read_text() == after


@pytest.mark.parametrize("text", ["not toml [", 'ui.layout = "old"\n'])
def test_save_layout_leaves_a_file_that_it_cannot_edit(tmp_path, text):
    path = tmp_path / "ui.toml"
    path.write_text(text)
    with pytest.raises(ValueError):
        save_layout(path, "new")
    assert path.read_text() == text


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
def recorded(tmp_path, named_setup):
    """Session and frames for each state of the done test of specs/ui.md, section 11."""
    cfg, frame = named_setup
    session = make_session(tmp_path, cfg, frame, teams=("orks",))
    frames = {"before the first frame": session.current()}
    live = settled(session, frame)
    frames["live"] = live
    frames["busy"] = replace(live, state=dict(live.state, busy=True))
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
    session.dispatch("board.click", x=x, y=y)
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
    win = MainWindow(session, FakeCamera(frame), layout_file(tmp_path, request.param))
    win.resize(1000, 700)
    win.show()
    yield win
    win.close()


@pytest.fixture
def idle_panel(qapp, tmp_path, recorded):
    session, frame, _ = recorded
    win = MainWindow(session, FakeCamera(frame), layout_file(tmp_path, "panel"))
    yield win
    win.close()


@pytest.fixture
def idle_inspector(qapp, tmp_path, recorded):
    session, frame, _ = recorded
    win = MainWindow(session, FakeCamera(frame), layout_file(tmp_path, "inspector"))
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
    window.current_layout.controls["snapshot.back"][1].click()
    wait_until(lambda: window.state["mode"] == "live")


def test_panel_quit_button_closes_the_window(qapp, tmp_path):
    cfg = ident_cfg()
    frame = paint_board(cfg, GAME)
    win = MainWindow(make_session(tmp_path, cfg, frame), FakeCamera(frame), layout_file(tmp_path, "panel"))
    win.show()
    win.start()
    wait_until(lambda: win.state is not None)
    win.current_layout.controls["app.quit"][1].click()
    assert win.worker.isFinished() and win.camera.released


@pytest.mark.parametrize("side, panel_first", [("left", True), ("right", False)])
def test_panel_side_option(qapp, tmp_path, recorded, side, panel_first):
    session, frame, _ = recorded
    path = tmp_path / "ui.toml"
    path.write_text(f'[ui]\nlayout = "panel"\n\n[layouts.panel]\npanel_side = "{side}"\n')
    win = MainWindow(session, FakeCamera(frame), path)
    assert (win.centralWidget().layout().itemAt(0).widget() is not win.video) is panel_first
    win.close()


# --- layout choice in the window ----------------------------------------------------


class OtherLayout(PanelLayout):
    name = "other"


def queued_notices(win: MainWindow) -> list[str]:
    return [win.worker.notices.get_nowait() for _ in range(win.worker.notices.qsize())]


@pytest.mark.parametrize(
    "text, wanted",
    [
        ('[ui]\nlayout = "no-such"\n', "unknown layout"),
        ("not toml [", "ui.toml not used"),
        ('[layouts.inspector]\npanel_side = "up"\n', "panel_side"),
        ('[layouts.inspector]\ndebug_open = "yes"\n', "debug_open"),
    ],
)
def test_bad_settings_do_not_stop_the_app(qapp, tmp_path, recorded, text, wanted):
    session, frame, _ = recorded
    path = tmp_path / "ui.toml"
    path.write_text(text)
    win = MainWindow(session, FakeCamera(frame), path)
    assert win.current_layout.name == DEFAULT_LAYOUT
    assert wanted in " ".join(queued_notices(win))
    win.close()


def test_a_missing_file_gives_no_notice_and_the_start_does_not_write_it(qapp, tmp_path, recorded):
    session, frame, _ = recorded
    path = tmp_path / "ui.toml"
    win = MainWindow(session, FakeCamera(frame), path)
    assert win.current_layout.name == DEFAULT_LAYOUT and queued_notices(win) == [] and not path.exists()
    win.close()


def test_settings_notice_reaches_the_state(qapp, tmp_path, named_setup):
    cfg, frame = named_setup
    path = tmp_path / "ui.toml"
    path.write_text('[ui]\nlayout = "no-such"\n')
    win = MainWindow(make_session(tmp_path, cfg, frame), FakeCamera(frame), path)
    win.show()
    win.start()
    wait_until(lambda: win.state is not None and win.state["notice"] is not None and "unknown layout" in win.state["notice"]["text"])
    win.close()


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
    assert load_ui_settings(window.ui_file).layout == "other"
    done = thread.done
    wait_until(lambda: thread.done == done and window.state["mode"] == "snapshot" and window.state["selected"] == 1 and window.current_layout.controls["names.accept"][1].isEnabled())
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


def test_l_key_goes_to_the_next_layout_and_saves_it(window):
    assert window.current_layout.name == "panel"
    press(window, "app.layout")
    assert isinstance(window.current_layout, InspectorLayout)
    assert load_ui_settings(window.ui_file).layout == "inspector" and window.layout_items["inspector"].isChecked()
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


def test_inspector_debug_opens_with_the_toggle_and_the_option(qapp, tmp_path, recorded):
    session, frame, _ = recorded
    path = tmp_path / "ui.toml"
    path.write_text('[layouts.inspector]\ndebug_open = true\npanel_side = "left"\n')
    win = MainWindow(session, FakeCamera(frame), path)
    win.show()
    layout = win.current_layout
    win._on_frame(recorded[2]["live"], 0)
    assert layout.debug.isVisible() and "Tags found: 4" in layout.debug.text()
    assert win.centralWidget().layout().itemAt(0).widget() is not win.video
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
    control_of(inspector, "snapshot.back").click()
    wait_until(lambda: inspector.state["mode"] == "live")


def test_inspector_shows_ring_buttons_and_the_distance(idle_inspector, recorded):
    layout = idle_inspector.current_layout
    idle_inspector._on_frame(recorded[2]["base selected"], 0)
    assert layout.distance.text() == "Click a second base to measure."
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
    wait_until(lambda: inspector.state["bases"][1]["confirmed"] and inspector.state["selected"] is None)
    assert inspector.state["bases"][1]["label"] == label
    wait_until(lambda: layout.stack.currentWidget() is layout.pages["snapshot"])


def test_inspector_operative_row_selects_the_base_and_esc_goes_back(inspector):
    open_names(inspector)
    layout = inspector.current_layout
    wait_until(lambda: layout.operatives.count() == 3)
    row_click(layout.operatives, 2)
    wait_until(lambda: inspector.state["selected"] == 2)
    press(inspector, "snapshot.back")
    wait_until(lambda: inspector.state["selected"] is None and inspector.state["mode"] == "snapshot")


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


def test_image_source_runs_the_live_app(monkeypatch, tmp_path, named_setup):
    cfg, frame = named_setup
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


def test_map_choice_is_saved_and_restored(qapp, tmp_path, named_setup):
    from battlesight.ui.settings import save_map

    from .test_live import write_map

    cfg, frame = named_setup
    write_map(tmp_path / "maps")
    path = layout_file(tmp_path, "inspector")
    save_map(path, "Test World", "Small")
    settings = load_ui_settings(path)
    assert (settings.layout, settings.map_terrain, settings.map_layout) == ("inspector", "Test World", "Small")
    win = MainWindow(make_session(tmp_path, cfg, frame), FakeCamera(frame), path)
    assert win.worker.commands.qsize() == 2  # restore: terrain type, then layout
    win.dispatch("map.layout", name=None)
    assert load_ui_settings(path).map_layout is None and load_ui_settings(path).map_terrain == "Test World"
    win.close()
