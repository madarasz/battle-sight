"""Layout `rosters`: a control bar on top, a game bar, the video, and a team
list each side.

The game bar between the control bar and the video holds the team choice of
each side (above its team list) and, in the centre, the turn (1 to 4) and the
Crit Op of the game.

The control bar has separate sections: View, Commands (with the Debug check box
at the bottom), Inspector and Reference. The inspector shows the selected operative
(name, crop, wounds, ring slider, name fix, remove a false base, shoot at it)
or the selected hatchway or breach point (open or closed). While "Shoot at" is
on, the right side of the inspector shows the board under the pointer and the name of the shooter
(the crop of the selected operative stays), a left click locks the shot ("Unlock" frees it), and OBSCURED and IN COVER show
what the terrain in the targeting lines gives. While "Locate operative" or "Locate terrain" waits for a click, it shows what to click and
the board under the pointer; the base that this finds opens with its name
list. While the name list is open, it takes the place of the wounds, ring
and distance. Each team list shows the operatives of one team, their wounds and
how they are identified; below it, the score counter of that side (VP
skulls, Tac Op, Primary, CP and the total). The name fix
is a list in the inspector, so the shell popup is off here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractButton, QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QStackedWidget, QToolButton, QVBoxLayout, QWidget

from ...live import ACTIONS, Action, shown_view
from ...colors import MESSAGE_NONE_COLOR, MODE_LIVE_COLOR, MODE_MINIMAP_COLOR, MODE_SNAPSHOT_COLOR, SHOT_IN_COVER_COLOR, SHOT_OBSCURED_COLOR
from ..parts import CandidateList, CopyList, GameControl, LightControl, MapPicker, OpeningControl, RingSlider, ScoreCounter, TeamRoster, VideoWidget, WoundsControl, action_control, aim_texts, distance_label, light_title, model_text, show_action_state

if TYPE_CHECKING:
    from . import Shell

ACTION_BY_ID = {a.id: a for a in ACTIONS}
TOP_HEIGHT = 175
GAME_BAR_STYLE = "#game_bar { border-top: 1px solid palette(mid); border-bottom: 1px solid palette(mid); }"
ROSTER_WIDTH = 310
CROP_SIZE = 96
OPENING_NAMES = {"hatchway": "Hatchway", "breach": "Breach point"}
CHANGE_TEXT = "Change operative"  # the list of the operatives and models, with their scores
CHANGE_ID_TEXT = "Change ID"  # the list of the copies of the base's type
MODE_BADGE = {"live": ("Live", MODE_LIVE_COLOR, "dot"), "snapshot": ("Snapshot", MODE_SNAPSHOT_COLOR, "pause"), "minimap": ("Minimap", MODE_MINIMAP_COLOR, "square")}


shown_mode = shown_view  # what the video shows: the minimap covers the live video and the snapshot


def operative_title(state: dict[str, Any]) -> str:
    """`Team - Operative (40mm)` of the selected base."""
    base = state["bases"][state["selected"]]
    size = f"({base['diameter_mm']:g}mm)"
    if base["label"] is not None:
        team, name = base["label"].partition("/")[0], base["name"]
        return f"{team} - {name} {size}" if name else f"{team} {size}"
    return f"Unknown operative {size}" if state["naming"] else f"Base {size}"


def roster_operative_title(state: dict[str, Any]) -> str:
    """`Team - Operative (40mm)` of the operative selected in a team list
    without a base; no size when the team file has none."""
    label = state["selected_operative"]
    team, _, name = label.partition("/")
    row = next((r for roster in state["rosters"] if roster is not None for r in roster["operatives"] if r["label"] == label), None)
    size = None if row is None else row["base_mm"]
    name = name if row is None else row["name"]
    return f"{team} - {name}" + ("" if size is None else f" ({size:g}mm)")


MEASURE_HINT = "Right click a second base to measure to."
SHOOT_HINT = "Point at the shooter, click to lock the shot. Push Cancel or Esc to stop."


def shooter_text(state: dict[str, Any]) -> str:
    """`Team - Operative` of the base under the pointer in "Shoot at". Empty when the pointer is not on a base."""
    shot = state["shot"]
    index = None if shot is None else shot["shooter"]
    if index is None or index >= len(state["bases"]):
        return ""
    base = state["bases"][index]
    if base["label"] is None:
        return "Unknown operative"
    team, name = base["label"].partition("/")[0], base["name"]
    return f"{team} - {name}" if name else team


def inspector_page(state: dict[str, Any]) -> str:
    if state["adding"] or state["locating_light"]:
        return "adding"
    if state["selected"] is not None:
        return "operative"
    if state["selected_operative"] is not None:
        return "roster_operative"
    if state["selected_light"] is not None:
        return "light"
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
        self._no_distance = False  # "Shoot at" or movement is on: the distance text is hidden
        self.fix_for: int | None = None  # base of the open "Change ID" list

    def build(self, shell: Shell, options: dict[str, Any]) -> QWidget:
        debug_open = options.get("debug_open", False)
        if not isinstance(debug_open, bool):
            raise ValueError(f"debug_open must be true or false, not {debug_open!r}")
        self.shell = shell
        root = QWidget()
        column = QVBoxLayout(root)
        column.setContentsMargins(4, 4, 4, 0)
        column.addWidget(self._top_bar(), 0)
        self.rosters = [TeamRoster(slot, shell.dispatch, team_on_top=False) for slot in (0, 1)]
        column.addWidget(self._game_bar(), 0)
        row = QHBoxLayout()
        self.counters = [ScoreCounter(slot, shell.dispatch) for slot in (0, 1)]
        sides = []
        for roster, counter in zip(self.rosters, self.counters):
            side = QWidget()
            side.setFixedWidth(ROSTER_WIDTH)
            side_column = QVBoxLayout(side)
            side_column.setContentsMargins(0, 0, 0, 0)
            side_column.setSpacing(3)  # little room between the team list buttons and the counter
            side_column.addWidget(roster, 1)
            side_column.addWidget(counter, 0)  # below the team list
            sides.append(side)
        self.video: VideoWidget = shell.new_video()
        row.addWidget(sides[0], 0)
        row.addWidget(self.video, 1)
        row.addWidget(sides[1], 0)
        column.addLayout(row, 1)
        self.debug_toggle.setChecked(debug_open)
        shell.show_debug(debug_open)
        return root

    # --- top bar ---------------------------------------------------------------------

    def _top_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(TOP_HEIGHT)
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(self._view_box(), 0)
        row.addWidget(self._commands_box(), 0)
        row.addWidget(self._inspector_box(), 1)
        row.addWidget(self._reference_box(), 0)
        return bar

    def _game_bar(self) -> QWidget:
        """The bar between the top bar and the video: the team choice of each
        side above its team list, the turn and the Crit Op in the centre."""
        bar = QWidget()
        bar.setObjectName("game_bar")
        bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bar.setStyleSheet(GAME_BAR_STYLE)
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(6)
        self.game = GameControl(self.shell.dispatch)
        left, right = (roster.team for roster in self.rosters)
        for team in (left, right):
            team.setFixedWidth(ROSTER_WIDTH)
        row.addWidget(left, 0)
        row.addStretch(1)
        row.addWidget(self.game, 0)
        row.addStretch(1)
        row.addWidget(right, 0)
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
        for k, action_id in enumerate(("view.top_down", "view.grid", "view.terrain", "view.zones", "view.keys", "view.light", "measure.walls", "view.marks")):
            toggles.addWidget(self._control(action_id), k // 4, k % 4)
        column.addLayout(toggles)
        column.addStretch(1)
        modes = QHBoxLayout()  # bottom row: what the video shows, one third each
        modes.setSpacing(0)
        for action_id in ("view.live", "view.snapshot", "view.minimap"):
            modes.addWidget(self._control(action_id), 1, Qt.AlignmentFlag.AlignHCenter)  # centred in its third
        column.addLayout(modes)
        return box

    def _commands_box(self) -> QGroupBox:
        box, column = self._section("Commands")
        column.addWidget(self._control("snapshot.name"))
        self.accept = self._control("names.accept")
        column.addWidget(self.accept)
        self.add_operative = self._control("base.add")
        self.add_operative.setToolTip("An operative has no base on the video: click the centre of its base")
        column.addWidget(self.add_operative)
        self.identify_terrains = self._control("light.locate")
        self.identify_terrains.setToolTip("Find the light terrain pieces of the layout on the board")
        column.addWidget(self.identify_terrains)
        self.locate_terrain = self._control("light.locate_at")
        self.locate_terrain.setToolTip("Click a light terrain piece on the video: the app finds which piece it is")
        column.addWidget(self.locate_terrain)
        column.addStretch(1)
        self.debug_toggle = _no_focus(QCheckBox("Debug"))
        self.debug_toggle.setToolTip("Debug values in the status bar")
        self.debug_toggle.toggled.connect(self.shell.show_debug)
        column.addWidget(self.debug_toggle)  # at the bottom of the section
        return box

    def _reference_box(self) -> QGroupBox:
        box, column = self._section("Reference")
        for action_id in ("reference.take", "reference.clear", "frame.save", "state.save"):
            column.addWidget(self._control(action_id))
        column.addStretch(1)
        self.reference_label = QLabel()
        column.addWidget(self.reference_label, 0, Qt.AlignmentFlag.AlignHCenter)  # at the bottom centre of the section
        return box

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
        self.change = _no_focus(QPushButton(CHANGE_TEXT))
        self.change.setCheckable(True)
        self.change.toggled.connect(lambda on: self._show_fix(self.change, on))
        header.addWidget(self.change, 0)
        self.change_id = _no_focus(QPushButton(CHANGE_ID_TEXT))  # only for a type with more than one copy in play
        self.change_id.setCheckable(True)
        self.change_id.toggled.connect(lambda on: self._show_fix(self.change_id, on))
        header.addWidget(self.change_id, 0)
        self.remove = self._control("base.remove")
        self.remove.setToolTip("Not an operative (terrain, board print): remove this base from the snapshot")
        header.addWidget(self.remove, 0)
        self.shoot = _no_focus(QPushButton("Shoot at"))  # "Cancel" while it is on
        self.shoot.clicked.connect(lambda: self.shell.dispatch("measure.shoot"))
        header.addWidget(self.shoot, 0)
        self.header = header  # "Remove ID" of a light terrain piece goes here too
        header.addStretch(1)
        self.close = _no_focus(QToolButton())
        self.close.setText("✕")
        self.close.setToolTip("Clear the selection")
        self.close.clicked.connect(lambda: self.shell.dispatch("base.deselect"))
        header.addWidget(self.close, 0, Qt.AlignmentFlag.AlignTop)
        column.addLayout(header)
        self.stack = QStackedWidget()
        for name, make in (("empty", self._empty_page), ("adding", self._adding_page), ("operative", self._operative_page), ("roster_operative", self._roster_operative_page), ("opening", self._opening_page), ("light", self._light_page)):
            self.pages[name] = page = QWidget()
            make(page)
            self.stack.addWidget(page)
        column.addWidget(self.stack, 1)
        return box

    def _empty_page(self, page: QWidget) -> None:
        column = QVBoxLayout(page)
        hint = QLabel("Click an operative or a hatchway on the board, or an operative in a team list.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {MESSAGE_NONE_COLOR}")
        column.addWidget(hint)
        column.addStretch(1)

    def _adding_page(self, page: QWidget) -> None:
        row = QHBoxLayout(page)
        row.setContentsMargins(0, 0, 0, 0)
        self.aim_crop = VideoWidget()
        self.aim_crop.setMinimumSize(80, 80)
        self.aim_crop.setFixedSize(CROP_SIZE, CROP_SIZE)
        row.addWidget(self.aim_crop, 0, Qt.AlignmentFlag.AlignTop)
        self.aim_hint = QLabel()
        self.aim_hint.setWordWrap(True)
        row.addWidget(self.aim_hint, 1, Qt.AlignmentFlag.AlignTop)

    def _operative_page(self, page: QWidget) -> None:
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)
        row = QHBoxLayout()
        outer.addLayout(row, 1)
        self.crop = VideoWidget()
        self.crop.setMinimumSize(80, 80)
        self.crop.setFixedSize(CROP_SIZE, CROP_SIZE)
        row.addWidget(self.crop, 0, Qt.AlignmentFlag.AlignTop)
        column = QVBoxLayout()
        column.setSpacing(2)
        self.model = QLabel()  # the model (counts_as variant) of the base, in italics; hidden for a type with one model
        font = self.model.font()
        font.setItalic(True)
        self.model.setFont(font)
        column.addWidget(self.model)
        self.details = QWidget()  # wounds, ring, movement and distance: hidden while the name list is open
        game = QVBoxLayout(self.details)
        game.setContentsMargins(0, 0, 0, 0)
        game.setSpacing(2)
        self.wounds = WoundsControl(self.shell.dispatch)
        game.addWidget(self.wounds)
        self.measure = QWidget()  # ring, movement and distance: hidden while "Shoot at" is on
        measure = QVBoxLayout(self.measure)
        measure.setContentsMargins(0, 0, 0, 0)
        measure.setSpacing(2)
        self.ring = RingSlider(self.shell.dispatch)
        measure.addWidget(self.ring)
        move_row = QHBoxLayout()
        move_row.addWidget(self._control("measure.move"))
        move_row.addSpacing(16)
        move_row.addWidget(self._control("measure.avoid_melee"))
        move_row.addStretch(1)
        measure.addLayout(move_row)
        game.addWidget(self.measure)
        self.shot = QWidget()  # what the terrain in the targeting lines gives
        shot = QVBoxLayout(self.shot)
        shot.setContentsMargins(0, 0, 0, 0)
        shot.setSpacing(2)
        flags = QHBoxLayout()
        self.obscured = QLabel("OBSCURED")
        self.in_cover = QLabel("IN COVER")
        for label, color in ((self.obscured, SHOT_OBSCURED_COLOR), (self.in_cover, SHOT_IN_COVER_COLOR)):
            font = label.font()
            font.setBold(True)
            font.setPointSizeF(font.pointSizeF() * 1.3)
            label.setFont(font)
            label.setStyleSheet(f"color: {color}")
            flags.addWidget(label, 0)
            flags.addSpacing(16)
        flags.addStretch(1)
        shot.addLayout(flags)
        self.shot_text = QLabel()
        self.shot_text.setWordWrap(True)
        shot.addWidget(self.shot_text)
        self.shot.setVisible(False)
        game.addWidget(self.shot)
        column.addWidget(self.details)
        self.candidates = CandidateList(self.shell.dispatch)
        self.candidates.setVisible(False)
        self.candidates.itemClicked.connect(lambda _item: self.change.setChecked(False))  # a name choice closes the list
        column.addWidget(self.candidates, 1)
        self.copies = CopyList(self.shell.dispatch)
        self.copies.setVisible(False)
        self.copies.itemClicked.connect(lambda _item: self.change_id.setChecked(False))
        column.addWidget(self.copies, 1)
        column.addStretch(0)
        row.addLayout(column, 1)
        self.aim = QWidget()  # "Shoot at": the board under the pointer and the name of the shooter, at the right
        aim = QVBoxLayout(self.aim)
        aim.setContentsMargins(0, 0, 0, 0)
        aim.setSpacing(2)
        self.shot_crop = VideoWidget()
        self.shot_crop.setMinimumSize(80, 80)
        self.shot_crop.setFixedSize(CROP_SIZE, CROP_SIZE)
        aim.addWidget(self.shot_crop, 0, Qt.AlignmentFlag.AlignHCenter)
        self.shooter = QLabel()
        self.shooter.setWordWrap(True)
        self.shooter.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.shooter.setFixedWidth(CROP_SIZE + 24)
        aim.addWidget(self.shooter)
        aim.addStretch(1)
        self.aim.setVisible(False)
        row.addWidget(self.aim, 0, Qt.AlignmentFlag.AlignTop)
        bottom = QHBoxLayout()  # last row, right of the crop: a left click locks the shot or the movement (status and "Unlock"); else the distance
        bottom.setContentsMargins(0, 0, 0, 0)
        self.lock_row = QWidget()
        lock = QHBoxLayout(self.lock_row)
        lock.setContentsMargins(0, 0, 0, 0)
        lock.setSpacing(10)
        self.lock_text = QLabel()
        lock.addWidget(self.lock_text)
        self.unlock = _no_focus(QPushButton("Unlock"))
        self.unlock.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.unlock.clicked.connect(lambda: self.shell.dispatch("measure.unlock"))
        lock.addWidget(self.unlock)
        self.lock_row.setVisible(False)
        bottom.addWidget(self.lock_row, 0)
        self.distance = QLabel()
        bottom.addWidget(self.distance, 0)
        bottom.addStretch(1)
        column.addLayout(bottom)

    def _show_fix(self, button: QPushButton, on: bool) -> None:
        """A list takes the place of the wounds, ring and distance: "Change
        operative" (the models and their scores) or "Change ID" (the copies).
        One list is open at a time. While it is open, its button is "Cancel"."""
        if on:
            (self.change_id if button is self.change else self.change).setChecked(False)
        self.change.setText("Cancel" if self.change.isChecked() else CHANGE_TEXT)
        self.change.setToolTip("Close the list, keep the operative" if self.change.isChecked() else "Choose another operative or model for this base")
        self.change_id.setText("Cancel" if self.change_id.isChecked() else CHANGE_ID_TEXT)
        self.change_id.setToolTip("Close the list, keep the operative" if self.change_id.isChecked() else "Choose which operative of the team list this base is (Boy #1, Boy #2, ...)")
        self.candidates.setVisible(self.change.isChecked())
        self.copies.setVisible(self.change_id.isChecked())
        self.details.setVisible(not (self.change.isChecked() or self.change_id.isChecked()))
        self._place_distance()

    def _roster_operative_page(self, page: QWidget) -> None:
        """An operative selected in a team list that has no base: only its wounds."""
        column = QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        self.roster_wounds = WoundsControl(self.shell.dispatch)
        column.addWidget(self.roster_wounds)
        column.addStretch(1)

    def _opening_page(self, page: QWidget) -> None:
        column = QVBoxLayout(page)
        self.opening = OpeningControl(self.shell.dispatch)
        column.addWidget(self.opening)
        column.addStretch(1)

    def _light_page(self, page: QWidget) -> None:
        column = QVBoxLayout(page)
        column.setContentsMargins(0, 0, 0, 0)
        self.light = LightControl(self.shell.dispatch, CROP_SIZE, beside=True, title_row=self.header)
        column.addWidget(self.light)
        column.addStretch(1)

    # --- state ---------------------------------------------------------------------

    def on_state(self, state: dict[str, Any]) -> None:
        for action, control in self.controls:
            show_action_state(control, action, state)
        self.accept.setVisible(state["unconfirmed"] > 0)
        self.map_picker.show_state(state)
        self.reference_label.setText(f"Reference: {'set' if state['reference_set'] else 'not set'}")
        for roster in self.rosters:
            roster.show_state(state)
        for counter in self.counters:
            counter.show_state(state)
        self.game.show_state(state)
        self.video.set_badge(*MODE_BADGE[shown_mode(state)])
        self._show_inspector(state)

    def _show_inspector(self, state: dict[str, Any]) -> None:
        page = inspector_page(state)
        self.stack.setCurrentWidget(self.pages[page])
        self.close.setVisible(page in ("operative", "roster_operative", "opening", "light"))
        self.change.setVisible(page == "operative")
        copies_shown = page == "operative" and bool(state["copy_choices"])
        self.change_id.setVisible(copies_shown)
        if not copies_shown and self.change_id.isChecked():
            self.change_id.setChecked(False)
        self.remove.setVisible(page == "operative")
        self.shoot.setVisible(page == "operative")
        if page != "light":
            self.light.remove.setVisible(False)
        self.shoot.setEnabled(ACTION_BY_ID["measure.shoot"].enabled(state))
        self.shoot.setText("Cancel" if state["shooting"] else "Shoot at")
        self.shoot.setToolTip("Stop the targeting lines (Esc)" if state["shooting"] else "Targeting lines from the pointer to this operative")
        if state["selected"] != self.fix_for:  # a new base starts with the name list closed
            self.fix_for = state["selected"]
            self.change_id.setChecked(False)
            self.change.setChecked(state["selected"] is not None and state["selected"] == state["new_base"])  # a found base opens the name list
        if state["shooting"]:  # the targeting lines close the lists
            self.change.setChecked(False)
            self.change_id.setChecked(False)
        if page == "empty":
            self.title.setText("Nothing selected")
        elif page == "adding":
            title, hint = aim_texts(state)
            self.title.setText(title)
            self.aim_hint.setText(hint)
            self.aim_crop.set_image(self.shell.selected_crop())
        elif page == "roster_operative":
            self.title.setText(roster_operative_title(state))
            self.roster_wounds.show_state(state)
        elif page == "opening":
            self.title.setText(OPENING_NAMES.get(state["selected_opening"]["kind"], state["selected_opening"]["kind"]))
            self.opening.show_state(state)
        elif page == "light":
            self.title.setText(light_title(state))
            self.light.show_state(state, self.shell.selected_crop())
        else:
            self.title.setText(operative_title(state))
            crop = self.shell.selected_crop()
            self.crop.setVisible(crop is not None)
            self.crop.set_image(crop)
            self.wounds.show_state(state)
            self.ring.show_state(state)
            distance = state["distance"]
            self.distance.setText(MEASURE_HINT if distance is None else distance_label(distance))
            self._show_shot(state)
            self._show_lock(state)
            self.change.setEnabled(bool(state["candidates"]))
            self.candidates.show_state(state)
            self.copies.show_state(state)
            model = model_text(state)
            self.model.setText(model)
            self.model.setVisible(bool(model))

    def _place_distance(self) -> None:
        """The distance text: hidden while a list takes its place, and with "Shoot at" or the movement measurement (no second base then)."""
        self.distance.setVisible(not (self._no_distance or self.change.isChecked() or self.change_id.isChecked()))

    def _show_lock(self, state: dict[str, Any]) -> None:
        """"Shot locked" or "Movement locked" with the "Unlock" button."""
        shot, move = state["shooting"] and state["shot_locked"], state["move_locked"]
        self.lock_row.setVisible(shot or move)
        self._no_distance = state["shooting"] or state["movement"]
        self._place_distance()
        self.lock_text.setText("Shot locked." if shot else "Movement locked.")
        self.unlock.setToolTip("Let the pointer move the " + ("shot" if shot else "movement") + " again")

    def _show_shot(self, state: dict[str, Any]) -> None:
        """While "Shoot at" is on: OBSCURED and IN COVER in place of the ring, movement and distance."""
        shooting, shot = state["shooting"], state["shot"]
        self.measure.setVisible(not shooting)
        self.shot.setVisible(shooting)
        self.aim.setVisible(shooting)  # the crop keeps its place when the pointer is off the board for a moment
        if not shooting:
            return
        self.shot_crop.set_image(self.shell.aim_crop())
        self.shooter.setText(shooter_text(state))
        self.obscured.setVisible(shot is not None and shot["obscured"])
        self.in_cover.setVisible(shot is not None and shot["in_cover"])
        if shot is None:
            self.shot_text.setText(SHOOT_HINT)
        elif shot["obscured"] or shot["in_cover"]:
            self.shot_text.setText("")
        elif shot.get("too_near"):
            self.shot_text.setText("Within 2 in of the target: no cover.")
        else:
            self.shot_text.setText("No terrain in the way." if shot["intervening"] == 0 else "Heavy terrain in the way, not obscured.")
