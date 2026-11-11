"""Layout `rosters`: a control bar on top, the video, and a team list each side.

The control bar has separate sections: View, Commands, Inspector, Reference
and Debug. The inspector shows the selected operative (name, crop, wounds,
ring slider, name fix, remove a false base) or the selected hatchway or breach point (open or
closed). Each team list shows the operatives of one team, their wounds and
how they are identified. The name fix
is a list in the inspector, so the shell popup is off here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractButton, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QStackedWidget, QToolButton, QVBoxLayout, QWidget

from ...live import ACTIONS, Action
from ..parts import COLORS, CandidateList, MapPicker, OpeningControl, RingSlider, TeamRoster, VideoWidget, WoundsControl, action_control, distance_label, show_action_state

if TYPE_CHECKING:
    from . import Shell

ACTION_BY_ID = {a.id: a for a in ACTIONS}
TOP_HEIGHT = 175
ROSTER_WIDTH = 240
CROP_SIZE = 96
OPENING_NAMES = {"hatchway": "Hatchway", "breach": "Breach point"}
MODE_BADGE = {"live": ("Live", COLORS["error"], "dot"), "snapshot": ("Snapshot", COLORS["info"], "pause"), "minimap": ("Minimap", COLORS["ok"], "square")}


def shown_mode(state: dict[str, Any]) -> str:
    """What the video shows: the minimap covers the live video and the snapshot."""
    return "minimap" if state["minimap"] else state["mode"]


def operative_title(state: dict[str, Any]) -> str:
    """`Team - Operative (40mm)` of the selected base."""
    base = state["bases"][state["selected"]]
    size = f"({base['diameter_mm']:g}mm)"
    if base["label"] is not None:
        team, _, name = base["label"].partition("/")
        return f"{team} - {name} {size}" if name else f"{team} {size}"
    return f"Unknown operative {size}" if state["naming"] else f"Base {size}"


def inspector_page(state: dict[str, Any]) -> str:
    if state["selected"] is not None:
        return "operative"
    return "opening" if state["selected_opening"] is not None else "empty"


def _no_focus(widget: QWidget) -> QWidget:
    widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    return widget


class RostersLayout:
    name: ClassVar[str] = "rosters"
    name_popup: ClassVar[bool] = False

    def __init__(self) -> None:
        self.controls: list[tuple[Action, QAbstractButton]] = []
        self.pages: dict[str, QWidget] = {}
        self.fix_for: int | None = None  # base of the open "Change ID" list

    def build(self, shell: Shell, options: dict[str, Any]) -> QWidget:
        debug_open = options.get("debug_open", False)
        if not isinstance(debug_open, bool):
            raise ValueError(f"debug_open must be true or false, not {debug_open!r}")
        self.shell = shell
        root = QWidget()
        column = QVBoxLayout(root)
        column.setContentsMargins(4, 4, 4, 0)
        column.addWidget(self._top_bar(debug_open), 0)
        row = QHBoxLayout()
        self.rosters = [TeamRoster(slot, shell.dispatch) for slot in (0, 1)]
        for roster in self.rosters:
            roster.setFixedWidth(ROSTER_WIDTH)
        self.video: VideoWidget = shell.new_video()
        row.addWidget(self.rosters[0], 0)
        row.addWidget(self.video, 1)
        row.addWidget(self.rosters[1], 0)
        column.addLayout(row, 1)
        shell.show_debug(debug_open)
        return root

    # --- top bar ---------------------------------------------------------------------

    def _top_bar(self, debug_open: bool) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(TOP_HEIGHT)
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(self._view_box(), 0)
        row.addWidget(self._commands_box(), 0)
        row.addWidget(self._inspector_box(), 1)
        row.addWidget(self._reference_box(), 0)
        row.addWidget(self._debug_box(debug_open), 0)
        return bar

    @staticmethod
    def _section(title: str) -> tuple[QGroupBox, QVBoxLayout]:
        """A titled section of the top bar, with small margins."""
        box = QGroupBox(title)
        column = QVBoxLayout(box)
        column.setContentsMargins(8, 4, 8, 6)
        column.setSpacing(4)
        return box, column

    def _control(self, action_id: str) -> QAbstractButton:
        action = ACTION_BY_ID[action_id]
        control = action_control(action, self.shell.dispatch)
        self.controls.append((action, control))
        return control

    def _view_box(self) -> QGroupBox:
        box, column = self._section("View")
        self.map_picker = MapPicker(self.shell.dispatch)
        for box_ in (self.map_picker.terrain, self.map_picker.layout_box):
            box_.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.map_picker.layout().setVerticalSpacing(1)
        column.addWidget(self.map_picker)
        column.addSpacing(4)
        toggles = QGridLayout()
        toggles.setVerticalSpacing(10)
        toggles.setHorizontalSpacing(14)
        for k, action_id in enumerate(("view.top_down", "view.grid", "view.terrain", "view.zones", "view.keys", "measure.walls", "view.minimap")):
            toggles.addWidget(self._control(action_id), k // 4, k % 4)
        column.addLayout(toggles)
        column.addStretch(1)
        return box

    def _commands_box(self) -> QGroupBox:
        box, column = self._section("Commands")
        column.addWidget(self._control("snapshot.name"))
        self.accept = self._control("names.accept")
        column.addWidget(self.accept)
        self.add_operative = _no_focus(QPushButton("Add operative"))
        self.add_operative.setEnabled(False)
        self.add_operative.setToolTip("Later: click where an operative is missing")
        column.addWidget(self.add_operative)
        column.addStretch(1)
        self.back = self._control("snapshot.back")
        column.addWidget(self.back)  # at the bottom of the section
        return box

    def _reference_box(self) -> QGroupBox:
        box, column = self._section("Reference")
        for action_id in ("reference.take", "reference.clear", "frame.save"):
            column.addWidget(self._control(action_id))
        column.addStretch(1)
        self.reference_label = QLabel()
        column.addWidget(self.reference_label, 0, Qt.AlignmentFlag.AlignHCenter)  # at the bottom centre of the section
        return box

    def _debug_box(self, debug_open: bool) -> QGroupBox:
        box, column = self._section("Debug")
        self.debug_toggle = _no_focus(QToolButton())
        self.debug_toggle.setCheckable(True)
        self.debug_toggle.setChecked(debug_open)
        self.debug_toggle.setText("Hide" if debug_open else "Show")
        self.debug_toggle.setToolTip("Debug values in the status bar")
        self.debug_toggle.toggled.connect(self._toggle_debug)
        column.addWidget(self.debug_toggle)
        column.addStretch(1)
        return box

    def _toggle_debug(self, on: bool) -> None:
        self.debug_toggle.setText("Hide" if on else "Show")
        self.shell.show_debug(on)

    # --- inspector ---------------------------------------------------------------------

    def _inspector_box(self) -> QGroupBox:
        box, column = self._section("Inspector")
        header = QHBoxLayout()
        self.title = QLabel()
        font = self.title.font()
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() * 1.15)
        self.title.setFont(font)
        header.addWidget(self.title, 0)
        self.change = _no_focus(QPushButton("Change ID"))
        self.change.setCheckable(True)
        self.change.toggled.connect(self._show_fix)
        header.addWidget(self.change, 0)
        self.remove = self._control("base.remove")
        self.remove.setToolTip("Not an operative (terrain, board print): remove this base from the snapshot")
        header.addWidget(self.remove, 0)
        header.addStretch(1)
        self.close = _no_focus(QToolButton())
        self.close.setText("✕")
        self.close.setToolTip("Clear the selection")
        self.close.clicked.connect(lambda: self.shell.dispatch("base.deselect"))
        header.addWidget(self.close, 0, Qt.AlignmentFlag.AlignTop)
        column.addLayout(header)
        self.stack = QStackedWidget()
        for name, make in (("empty", self._empty_page), ("operative", self._operative_page), ("opening", self._opening_page)):
            self.pages[name] = page = QWidget()
            make(page)
            self.stack.addWidget(page)
        column.addWidget(self.stack, 1)
        return box

    def _empty_page(self, page: QWidget) -> None:
        column = QVBoxLayout(page)
        hint = QLabel("Click an operative or a hatchway on the board, or an operative in a team list.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {COLORS['none']}")
        column.addWidget(hint)
        column.addStretch(1)

    def _operative_page(self, page: QWidget) -> None:
        row = QHBoxLayout(page)
        row.setContentsMargins(0, 0, 0, 0)
        self.crop = VideoWidget()
        self.crop.setMinimumSize(80, 80)
        self.crop.setFixedSize(CROP_SIZE, CROP_SIZE)
        row.addWidget(self.crop, 0, Qt.AlignmentFlag.AlignTop)
        column = QVBoxLayout()
        column.setSpacing(2)
        self.wounds = WoundsControl(self.shell.dispatch)
        column.addWidget(self.wounds)
        self.ring = RingSlider(self.shell.dispatch)
        column.addWidget(self.ring)
        column.addWidget(self._control("measure.move"))
        self.distance = QLabel()
        column.addWidget(self.distance)
        self.candidates = CandidateList(self.shell.dispatch)
        self.candidates.setVisible(False)
        column.addWidget(self.candidates, 1)
        column.addStretch(0)
        row.addLayout(column, 1)

    def _show_fix(self, on: bool) -> None:
        self.candidates.setVisible(on)

    def _opening_page(self, page: QWidget) -> None:
        column = QVBoxLayout(page)
        self.opening = OpeningControl(self.shell.dispatch)
        column.addWidget(self.opening)
        column.addStretch(1)

    # --- state ---------------------------------------------------------------------

    def on_state(self, state: dict[str, Any]) -> None:
        for action, control in self.controls:
            show_action_state(control, action, state)
        self.accept.setVisible(state["unconfirmed"] > 0)
        self.back.setVisible(state["mode"] == "snapshot")
        self.map_picker.show_state(state)
        self.reference_label.setText(f"Reference: {'set' if state['reference_set'] else 'not set'}")
        for roster in self.rosters:
            roster.show_state(state)
        self.video.set_badge(*MODE_BADGE[shown_mode(state)])
        self._show_inspector(state)

    def _show_inspector(self, state: dict[str, Any]) -> None:
        page = inspector_page(state)
        self.stack.setCurrentWidget(self.pages[page])
        self.close.setVisible(page != "empty")
        self.change.setVisible(page == "operative")
        self.remove.setVisible(page == "operative")
        if state["selected"] != self.fix_for:  # a new base starts with the name list closed
            self.fix_for = state["selected"]
            self.change.setChecked(False)
        if page == "empty":
            self.title.setText("Nothing selected")
        elif page == "opening":
            self.title.setText(OPENING_NAMES.get(state["selected_opening"]["kind"], state["selected_opening"]["kind"]))
            self.opening.show_state(state)
        else:
            self.title.setText(operative_title(state))
            crop = self.shell.selected_crop()
            self.crop.setVisible(crop is not None)
            self.crop.set_image(crop)
            self.wounds.show_state(state)
            self.ring.show_state(state)
            distance = state["distance"]
            self.distance.setText("Click a second base to measure." if distance is None else distance_label(distance))
            self.change.setEnabled(bool(state["candidates"]))
            self.candidates.show_state(state)
