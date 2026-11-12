"""The window of the live app: worker thread, keys, generated menu, status.

The worker owns the camera and the `LiveSession`. The GUI thread only sends
commands to its queue and shows the `LiveFrame`s that it gets back.
"""

from __future__ import annotations

import queue
import signal
import sys
import threading
import traceback
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QPoint, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QMenu, QMessageBox

from ..live import ACTIONS, Action, LiveFrame, LiveSession
from .layouts import DEFAULT_LAYOUT, LAYOUTS, Layout
from .parts import COLORS, DRIFT_LEVEL, VideoWidget, action_group, candidate_text, debug_line, distance_label
from .settings import DEFAULT_UI_FILE, load_ui_settings, save_layout, save_map

QT_KEYS = {"space": "Space", "Esc": "Esc"}


def status_text(state: dict[str, Any]) -> tuple[str, str]:
    """Status line and its level, from the core state."""
    if not state["camera"]:
        return "camera down: waiting for it to come back", "error"
    drift = state["drift"]
    if drift is None:
        return "waiting for the first frame", "warn"
    parts = [drift["text"], f"{state['lens']} lens"]
    if state.get("resolution"):
        w, h = state["resolution"]
        parts.append(f"{w}x{h}")
    if state["busy"]:
        parts.append("working ...")
    elif state["mode"] == "live":
        pass
    elif state["snapshot_ok"] is False:
        parts.append("snapshot failed")
    elif state["naming"]:
        named = sum(b["label"] is not None for b in state["bases"])
        parts.append(f"{named}/{len(state['bases'])} named")
    else:
        parts.append(f"{len(state['bases'])} bases")
    if state["ring_in"] is not None:
        parts.append(f"ring {state['ring_in']} in")
    if state.get("move") is not None:
        move = state["move"]
        steps = "" if len(move["increments"]) < 2 else " (" + " + ".join(str(c) for c in move["increments"]) + ")"
        parts.append(f'move {move["in"]}"{steps}')
    if state["distance"] is not None:
        parts.append(distance_label(state["distance"]).replace(":", "", 1).lower())
    return "  |  ".join(parts), DRIFT_LEVEL[drift["status"]]


def help_text() -> str:
    rows = [f"{a.key}\t{a.label}" for a in ACTIONS if a.key]
    rows.append("click\ta base: select it")
    rows.append("right click\ta second base: measure the distance")
    rows.append("click\ta hatchway or a breach point: select it")
    rows.append("right click\ta base after space: choose its name")
    return "\n".join(rows)


class Worker(QThread):
    """Reads the camera and runs the session. All calls to the session happen
    on this thread. Commands run between frames, in the order sent."""

    frame_ready = Signal(object, int)  # LiveFrame, number of commands done

    def __init__(self, session: LiveSession, camera):
        super().__init__()
        self.session = session
        self.camera = camera
        self.commands: queue.SimpleQueue[tuple[str, dict]] = queue.SimpleQueue()
        self.notices: queue.SimpleQueue[str] = queue.SimpleQueue()
        self.sent = 0  # GUI thread only
        self.done = 0  # worker thread only
        self.display_width = 0
        self._stop_event = threading.Event()
        self._gui_free = threading.Event()  # the GUI took the last frame
        self._gui_free.set()

    def submit(self, action_id: str, **args) -> int:
        """Queue one action. Returns its number. Call it from the GUI thread."""
        self.sent += 1
        self.commands.put((action_id, args))
        return self.sent

    def notify(self, text: str) -> None:
        """Show an error notice. The session gets it on this thread. Call it from the GUI thread."""
        self.notices.put(text)

    def gui_free(self) -> None:
        self._gui_free.set()

    def stop(self) -> None:
        self._stop_event.set()

    def _send_now(self, live: LiveFrame) -> None:
        self.frame_ready.emit(live, self.done)

    def _run_commands(self) -> None:
        while not self.notices.empty():
            self.session.notice.error(self.notices.get_nowait())
        while True:
            try:
                action_id, args = self.commands.get_nowait()
            except queue.Empty:
                return
            try:
                self.session.dispatch(action_id, **args)
            except Exception as exc:  # one bad command must not stop the video
                traceback.print_exc()
                self.session.notice.error(f"{action_id} failed: {exc}")
            self.done += 1

    def run(self) -> None:
        session = self.session
        session.on_busy = self._send_now
        while not self._stop_event.is_set():
            self._run_commands()
            if self.display_width and self.display_width != session.win.display_width:
                session.set_display_width(self.display_width)
            try:
                frame = self.camera.read()
                live = session.step(frame)
            except Exception as exc:
                traceback.print_exc()
                session.notice.error(f"frame failed: {exc}")
                self.msleep(100)
                continue
            if frame is None:
                self.msleep(100)
            if self._gui_free.is_set():  # drop frames while the GUI is behind
                self._gui_free.clear()
                self._send_now(live)


CHECK_ACTIONS = frozenset(a.id for a in ACTIONS if a.checked is not None)  # shown as check boxes


class MainWindow(QMainWindow):
    """Status bar, menu and keys, for all layouts. The layout is the central widget."""

    closed = Signal()

    def __init__(self, session: LiveSession, camera, ui_file: str | Path = DEFAULT_UI_FILE):
        super().__init__()
        self.setWindowTitle("Battle Sight")
        self.camera = camera
        self.worker = Worker(session, camera)
        self.state: dict[str, Any] | None = None
        self.last_frame: LiveFrame | None = None
        self.status_label = QLabel()
        self.notice_label = QLabel()
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.notice_label)
        self.debug_label = QLabel()  # a layout turns it on with show_debug
        self.debug_label.setStyleSheet(f"color: {COLORS['none']}")
        self.debug_label.setVisible(False)
        self.statusBar().addPermanentWidget(self.debug_label)
        self.menu_actions: dict[str, tuple[Action, QAction]] = {}
        self.layout_items: dict[str, QAction] = {}
        self.shortcuts: dict[str, QShortcut] = {}
        self._build_menu()
        self._build_keys()
        self.popup: QMenu | None = None
        self._pending_click: tuple[int, QPoint] | None = None  # command number, global position
        self._check_sent = 0  # command number of the last check box action
        self._hover_sent: int | None = None  # command number of the pointer position that the worker has
        self._hover_next: tuple | None = None  # (position or None,) to send after it
        self.worker.frame_ready.connect(self._on_frame)
        self.ui_file = Path(ui_file)
        self.ui_settings = load_ui_settings(self.ui_file)
        if self.ui_settings.problem:
            self.worker.notify(self.ui_settings.problem)
        self.video: VideoWidget | None = None
        self.current_layout: Layout | None = None
        self._install_layout(self.ui_settings.layout or DEFAULT_LAYOUT)
        self._restore_map()

    def _restore_map(self) -> None:
        """Choose the terrain map of the last run. The commands run before the first frame."""
        terrain, layout = self.ui_settings.map_terrain, self.ui_settings.map_layout
        self.map_choice = [terrain, layout]
        if terrain:
            self.worker.submit("map.terrain_type", name=terrain)
            if layout:
                self.worker.submit("map.layout", name=layout)

    def _save_map(self, action_id: str, name: str | None) -> None:
        """Remember the terrain type and layout that the user chose."""
        if action_id == "map.terrain_type":
            self.map_choice = [name, None if name != self.map_choice[0] else self.map_choice[1]]
        else:
            self.map_choice[1] = name
        try:
            save_map(self.ui_file, *self.map_choice)
        except (OSError, ValueError) as exc:
            self.worker.notify(f"map choice not saved: {exc}")

    def start(self) -> None:
        self.worker.start()

    # --- layouts -----------------------------------------------------------------------

    def new_video(self) -> VideoWidget:
        """The video widget of the layout that is being built."""
        self.video = VideoWidget()
        self.video.clicked.connect(self._on_click)
        self.video.right_clicked.connect(self._on_right_click)
        self.video.hovered.connect(self._on_hover)
        self.video.width_changed.connect(self._on_width)  # the first resize sends the width
        return self.video

    def show_debug(self, on: bool) -> None:
        """Show the debug values at the right end of the status bar."""
        self.debug_label.setVisible(on)
        if on and self.state is not None:
            self.debug_label.setText(debug_line(self.state))

    def selected_crop(self) -> np.ndarray | None:
        return None if self.last_frame is None else self.last_frame.crop

    def next_layout(self) -> None:
        names = list(LAYOUTS)
        self.set_layout(names[(names.index(self.current_layout.name) + 1) % len(names)])

    def set_layout(self, name: str) -> None:
        """Change the layout and save the choice. The worker, the camera and the game state stay."""
        self._install_layout(name)
        try:
            save_layout(self.ui_file, self.current_layout.name)
        except (OSError, ValueError) as exc:
            self.worker.notify(f"layout choice not saved: {exc}")

    def _install_layout(self, name: str) -> None:
        if name not in LAYOUTS:
            self.worker.notify(f"unknown layout {name!r}: using {DEFAULT_LAYOUT}")
            name = DEFAULT_LAYOUT
        if self.popup is not None:
            self.popup.close()
            self.popup = None
        options = self.ui_settings.options_for(name)
        self.show_debug(False)  # the new layout turns it on again when it wants it
        layout = LAYOUTS[name]()
        try:
            root = layout.build(self, options)
        except ValueError as exc:
            self.worker.notify(f"layout {name}: {exc}: using its defaults")
            layout = LAYOUTS[name]()
            root = layout.build(self, {})
        self.current_layout = layout
        self.setCentralWidget(root)  # Qt deletes the old layout widgets
        if name in self.layout_items:
            self.layout_items[name].setChecked(True)
        if self.last_frame is not None:
            self._show_frame(self.last_frame)

    # --- menu, keys, actions -----------------------------------------------------------

    def _build_menu(self) -> None:
        menus: dict[str, QMenu] = {}
        for action in ACTIONS:
            if action.args:
                continue
            title = action_group(action)
            menu = menus.get(title) or menus.setdefault(title, self.menuBar().addMenu(title))
            # The key is only text here: a QShortcut does the key, so that no two shortcuts compete.
            item = menu.addAction(f"{action.label}  [{action.key}]" if action.key else action.label)
            item.setCheckable(action.checked is not None)
            item.triggered.connect(partial(self.trigger, action))
            self.menu_actions[action.id] = (action, item)
        chooser = menus["View"].addMenu("Layout")
        group = QActionGroup(self)
        for name in LAYOUTS:
            item = group.addAction(chooser.addAction(name))
            item.setCheckable(True)
            item.triggered.connect(lambda _checked=False, name=name: self.set_layout(name))
            self.layout_items[name] = item

    def _build_keys(self) -> None:
        for action in ACTIONS:
            if action.key:
                shortcut = QShortcut(QKeySequence(QT_KEYS.get(action.key, action.key)), self)
                shortcut.activated.connect(partial(self.trigger, action))
                self.shortcuts[action.id] = shortcut

    def trigger(self, action: Action, *_checked) -> None:
        self.dispatch(action.id)

    def dispatch(self, action_id: str, **args) -> int | None:
        """Do one action. Returns its command number. Returns None when the
        front end did the action, or when a snapshot is running and the
        command is dropped."""
        if action_id == "app.quit":
            self.close()
        elif action_id == "app.layout":
            self.next_layout()
        elif action_id == "app.help":
            QMessageBox.information(self, "Keys", help_text())
        elif not self._busy():
            if action_id in ("map.terrain_type", "map.layout"):
                self._save_map(action_id, args.get("name"))
            number = self.worker.submit(action_id, **args)
            if action_id in CHECK_ACTIONS:
                self._check_sent = number
            return number
        return None

    def _busy(self) -> bool:
        """A snapshot is running. New commands would only pile up behind it."""
        return self.state is not None and self.state["busy"]

    def _on_width(self, width: int) -> None:
        self.worker.display_width = width

    def _on_click(self, x: float, y: float, _global_pos: QPoint) -> None:
        self.dispatch("board.click", x=x, y=y)

    def _on_right_click(self, x: float, y: float, global_pos: QPoint) -> None:
        number = self.dispatch("board.select", x=x, y=y)
        if number is not None:
            self._pending_click = number, global_pos

    def _on_hover(self, at) -> None:
        """The pointer moved. The movement measurement and "Locate operative"
        (the board under the pointer) use it. One position is with the worker
        at a time; the newest waits."""
        state = self.state
        if state is None:
            return
        moving = state.get("movement") and state["ring_in"] is not None and state["selected"] is not None
        if not moving and not state.get("adding"):
            return
        self._hover_next = (at,)
        self._send_hover()

    def _send_hover(self) -> None:
        if self._hover_sent is not None or self._hover_next is None:
            return
        (at,) = self._hover_next
        self._hover_next = None
        x, y = (None, None) if at is None else at
        self._hover_sent = self.dispatch("board.hover", x=x, y=y)

    def _on_frame(self, live: LiveFrame, done: int) -> None:
        try:
            # A frame from before a check box command has the old value: it
            # must not put the box back for a moment.
            self._show_frame(live, controls=done >= self._check_sent)
            if self._hover_sent is not None and done >= self._hover_sent:
                self._hover_sent = None
                self._send_hover()
            if self._pending_click is not None and done >= self._pending_click[0]:
                if self.current_layout.name_popup and live.state.get("right_click") == "name":
                    self._open_name_popup(live.state, self._pending_click[1])
                self._pending_click = None
        finally:  # the worker sends no more frames until this call
            self.worker.gui_free()

    def _show_frame(self, live: LiveFrame, controls: bool = True) -> None:
        self.last_frame = live
        self.state = state = live.state
        self.video.set_image(live.image)
        self._show_status(state)
        if not controls:
            return
        for action, item in self.menu_actions.values():
            item.setEnabled(action.enabled(state))
            if action.checked is not None:
                item.setChecked(action.checked(state))
        self.current_layout.on_state(state)

    def _show_status(self, state: dict[str, Any]) -> None:
        text, level = status_text(state)
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {COLORS[level]}")
        notice = state["notice"]
        self.notice_label.setText("" if notice is None else notice["text"])
        if self.debug_label.isVisible():
            self.debug_label.setText(debug_line(state))
        if notice is not None:
            self.notice_label.setStyleSheet(f"color: {COLORS[notice['level']]}")

    def _open_name_popup(self, state: dict[str, Any], global_pos: QPoint) -> None:
        index = state["selected"]
        if index is None or not state["candidates"]:
            return
        menu = QMenu(self)
        for candidate in state["candidates"]:
            menu.addAction(candidate_text(candidate)).triggered.connect(partial(self._choose_name, candidate["label"], index))
        self.popup = menu
        menu.popup(global_pos)

    def _choose_name(self, label: str, index: int, *_checked) -> None:
        self.worker.submit("base.choose", label=label, index=index)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.popup is not None:
            self.popup.close()
        self.worker.stop()
        self.worker.wait()
        self.camera.release()
        event.accept()
        self.closed.emit()


def run_window(session: LiveSession, camera, ui_file: str | Path = DEFAULT_UI_FILE) -> int:
    """Open the window and run until it closes. Returns the exit code."""
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = MainWindow(session, camera, ui_file)
    # Not app.quit: on macOS it goes through NSApp terminate, and the event loop sometimes stays.
    window.closed.connect(lambda: QTimer.singleShot(0, app.exit))
    screen = app.primaryScreen().availableGeometry()
    window.resize(int(screen.width() * 0.85), int(screen.height() * 0.85))
    window.show()
    window.start()
    previous = signal.signal(signal.SIGINT, lambda *_: window.close())  # Ctrl-C in the Terminal
    tick = QTimer()  # lets Python see the signal while Qt runs
    tick.timeout.connect(lambda: None)
    tick.start(200)
    try:
        return app.exec()
    finally:
        tick.stop()
        signal.signal(signal.SIGINT, previous)
