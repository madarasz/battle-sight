"""Layout `inspector`: the video and a panel that follows the state.

The top part (status and view toggles) stays. Below it one page shows: live,
snapshot, or selected base. The name fix is a list in the panel, with the crop
of the base and the scores, so the shell popup is off here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractButton, QHBoxLayout, QLabel, QStackedWidget, QToolButton, QVBoxLayout, QWidget

from ...live import ACTIONS, Action
from ..parts import CandidateList, DebugView, OperativeList, StatusView, VideoWidget, action_control, distance_text, panel_side, ring_controls, show_action_state

if TYPE_CHECKING:
    from . import Shell

ACTION_BY_ID = {a.id: a for a in ACTIONS}
PANEL_WIDTH = 300
CROP_HEIGHT = 220


def page_for(state: dict[str, Any]) -> str:
    if state["mode"] == "live":
        return "live"
    return "snapshot" if state["selected"] is None else "selected"


def summary_text(state: dict[str, Any]) -> str:
    if state["snapshot_ok"] is False:
        return "Snapshot failed"
    bases = state["bases"]
    if not state["naming"]:
        return f"{len(bases)} bases"
    named = sum(b["label"] is not None for b in bases)
    return f"{named}/{len(bases)} named, {state['unconfirmed']} not confirmed"


class InspectorLayout:
    name: ClassVar[str] = "inspector"
    name_popup: ClassVar[bool] = False

    def __init__(self) -> None:
        self.controls: list[tuple[Action, QAbstractButton]] = []  # one action can be on two pages
        self.pages: dict[str, QWidget] = {}

    def build(self, shell: Shell, options: dict[str, Any]) -> QWidget:
        side = panel_side(options)
        debug_open = options.get("debug_open", False)
        if not isinstance(debug_open, bool):
            raise ValueError(f"debug_open must be true or false, not {debug_open!r}")
        self.shell = shell
        root = QWidget()
        row = QHBoxLayout(root)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(shell.new_video(), 1)
        row.insertWidget(0 if side == "left" else 1, self._panel(debug_open), 0)
        return root

    def _panel(self, debug_open: bool) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(PANEL_WIDTH)
        column = QVBoxLayout(panel)
        self.status = StatusView()
        column.addWidget(self.status)
        self._add_controls(column, "view.top_down", "view.grid", "view.tags")
        self.stack = QStackedWidget()
        for name, make in (("live", self._live_page), ("snapshot", self._snapshot_page), ("selected", self._selected_page)):
            self.pages[name] = page = QWidget()
            make(QVBoxLayout(page))
            self.stack.addWidget(page)
        column.addWidget(self.stack, 1)
        self.debug_toggle = QToolButton()
        self.debug_toggle.setText("Debug")
        self.debug_toggle.setCheckable(True)
        self.debug_toggle.setChecked(debug_open)
        self.debug_toggle.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.debug = DebugView()
        self.debug.setVisible(debug_open)
        self.debug_toggle.toggled.connect(self.debug.setVisible)
        column.addWidget(self.debug_toggle)
        column.addWidget(self.debug)
        return panel

    def _add_controls(self, column: QVBoxLayout, *action_ids: str) -> None:
        for action_id in action_ids:
            action = ACTION_BY_ID[action_id]
            control = action_control(action, self.shell.dispatch)
            column.addWidget(control)
            self.controls.append((action, control))

    def _live_page(self, column: QVBoxLayout) -> None:
        self._add_controls(column, "snapshot.name", "snapshot.bases")
        self.reference_label = QLabel()
        column.addWidget(self.reference_label)
        self._add_controls(column, "reference.take", "reference.clear", "frame.save")
        column.addStretch(1)

    def _snapshot_page(self, column: QVBoxLayout) -> None:
        self.summary = QLabel()
        column.addWidget(self.summary)
        self.operatives = OperativeList(self.shell.dispatch)
        column.addWidget(self.operatives, 1)
        self._add_controls(column, "names.accept", "names.undo", "snapshot.name", "snapshot.bases", "frame.save", "snapshot.back")

    def _selected_page(self, column: QVBoxLayout) -> None:
        self.base_title = QLabel()
        column.addWidget(self.base_title)
        self.crop = VideoWidget()
        self.crop.setMinimumSize(100, 100)
        self.crop.setFixedHeight(CROP_HEIGHT)
        column.addWidget(self.crop)
        grid, rings = ring_controls(self.shell.dispatch)
        column.addWidget(grid)
        self.controls.extend(rings)
        self.distance = QLabel()
        self.distance.setWordWrap(True)
        column.addWidget(self.distance)
        self.candidates = CandidateList(self.shell.dispatch)
        column.addWidget(self.candidates, 1)
        self._add_controls(column, "snapshot.back")

    def on_state(self, state: dict[str, Any]) -> None:
        for action, control in self.controls:
            show_action_state(control, action, state)
        self.status.show_state(state)
        self.debug.show_state(state)
        page = page_for(state)
        self.stack.setCurrentWidget(self.pages[page])
        if page == "live":
            self.reference_label.setText(f"Reference: {'set' if state['reference_set'] else 'not set'}")
        elif page == "snapshot":
            self.summary.setText(summary_text(state))
            self.operatives.show_state(state)
        else:
            base = state["bases"][state["selected"]]
            self.base_title.setText(f"{base['diameter_mm']:g} mm base")
            crop = self.shell.selected_crop()
            self.crop.setVisible(crop is not None)
            self.crop.set_image(crop)
            distance = state["distance"]
            self.distance.setText("Click a second base to measure." if distance is None else f"Rim to rim: {distance_text(distance)}")
            self.candidates.show_state(state)
