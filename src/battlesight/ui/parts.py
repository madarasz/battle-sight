"""Qt widgets that all layouts share."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QMouseEvent, QPainter, QPainterPath, QPixmap
from PySide6.QtWidgets import QAbstractButton, QApplication, QCheckBox, QStyledItemDelegate, QStyleOptionViewItem, QComboBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QSlider, QSpinBox, QStyle, QToolButton, QVBoxLayout, QWidget

from ..live import ACTIONS, TEAM_COLORS, Action
from ..score import CATEGORIES, CATEGORY_MAX, CATEGORY_NAMES, PRIMARIES, SKULLS_PER_TP, TURNING_POINTS

ACTION_GROUPS = {"wounds": "Game", "score": "Game", "operative": "Snapshot", "opening": "Game", "team": "View", "map": "View", "snapshot": "Snapshot", "base": "Snapshot", "names": "Snapshot", "frame": "Snapshot", "ring": "Measure", "measure": "Measure", "view": "View", "reference": "Reference", "app": "Help"}
RING_COLUMNS = 5
COLORS = {"none": "#8a8a8a", "ok": "#2e9e4f", "done": "#2e9e4f", "warn": "#c98a00", "hint": "#c98a00", "error": "#d03a3a", "info": "#3d8bfd"}
DRIFT_LEVEL = {"OK": "ok", "NO_REFERENCE": "warn", "MOVED": "warn", "CARD_MOVED": "error", "LOST": "error"}


def action_group(action: Action) -> str:
    """Title of the menu or panel group of an action."""
    return ACTION_GROUPS[action.id.split(".")[0]]


def action_control(action: Action, dispatch: Callable[[str], Any]) -> QAbstractButton:
    """A check box or a push button for an action without arguments.

    It never takes the focus, or `space` pushes it instead of the shortcut."""
    control = QCheckBox(action.label) if action.checked is not None else QPushButton(action.label)
    control.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    if action.key:
        control.setToolTip(f"Key: {action.key}")
    control.clicked.connect(lambda *_checked: dispatch(action.id))
    return control


def ring_controls(dispatch: Callable[[str], Any]) -> tuple[QWidget, list[tuple[Action, QAbstractButton]]]:
    """A compact grid with one control per ring action, and the controls."""
    box = QWidget()
    grid = QGridLayout(box)
    grid.setContentsMargins(0, 0, 0, 0)
    controls = []
    for k, action in enumerate(a for a in ACTIONS if a.id.startswith("ring.")):
        control = action_control(action, dispatch)
        control.setText(action.key if action.checked is not None else "Clear")
        grid.addWidget(control, k // RING_COLUMNS, k % RING_COLUMNS)
        controls.append((action, control))
    return box, controls


def show_action_state(control: QAbstractButton, action: Action, state: dict[str, Any]) -> None:
    control.setEnabled(action.enabled(state))
    if action.checked is not None:
        control.setChecked(action.checked(state))


def panel_side(options: dict[str, Any]) -> str:
    """The `panel_side` option of a layout. A bad value is a ValueError."""
    side = options.get("panel_side", "right")
    if side not in ("left", "right"):
        raise ValueError(f"panel_side must be left or right, not {side!r}")
    return side


def percent(score: float) -> str:
    """An identification score as a percentage, for example `78%`."""
    return f"{score:.0%}"


def candidate_text(candidate: dict[str, Any]) -> str:
    score = candidate["score"]
    return candidate["label"] if score is None else f"{candidate['label']}   {percent(score)}"


def distance_text(distance: dict[str, Any]) -> str:
    return "no path" if distance["in"] is None else f'{distance["in"]:.2f}"'


def distance_label(distance: dict[str, Any]) -> str:
    """`Rim to rim: 5.25"`, or `Around walls: 6.10"` for a distance measured around the walls."""
    return f'{"Around walls" if distance.get("around") else "Rim to rim"}: {distance_text(distance)}'


def debug_text(state: dict[str, Any]) -> str:
    debug = state["debug"]

    def value(key: str, fmt: str, unit: str = "") -> str:
        return "-" if debug.get(key) is None else f"{debug[key]:{fmt}}{unit}"

    return "\n".join(
        (
            f"Tags found: {value('tags', 'd')}",
            f"Fit error: {value('fit_max_mm', '.2f', ' mm')}",
            f"Frames: {value('fps', '.1f', ' per s')}",
            f"Tag detection: {value('detect_ms', '.0f', ' ms')}",
        )
    )


def debug_line(state: dict[str, Any]) -> str:
    """The debug values on one line, for the status bar."""
    debug = state["debug"]

    def value(key: str, fmt: str, unit: str = "") -> str:
        return "-" if debug.get(key) is None else f"{debug[key]:{fmt}}{unit}"

    return f"fit {value('fit_max_mm', '.2f', ' mm')}  |  {value('fps', '.1f', ' fps')}  |  detect {value('detect_ms', '.0f', ' ms')}"


NO_LAYOUT = "None"


class MapPicker(QWidget):
    """Two drop-down lists: terrain type and map layout ("None" = no map).
    They fill again only when their lists change, and only a choice of the
    user sends an action."""

    def __init__(self, dispatch: Callable[..., Any], parent: QWidget | None = None):
        super().__init__(parent)
        self.dispatch = dispatch
        self.shown: Any = None
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        self.terrain = QComboBox()
        self.layout_box = QComboBox()
        for box in (self.terrain, self.layout_box):
            box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            box.view().setFocusPolicy(Qt.FocusPolicy.NoFocus)
        form.addRow("Terrain type", self.terrain)
        form.addRow("Layout", self.layout_box)
        self.terrain.activated.connect(lambda i: self.dispatch("map.terrain_type", name=self.terrain.itemText(i)))
        self.layout_box.activated.connect(lambda i: self.dispatch("map.layout", name=None if i == 0 else self.layout_box.itemText(i)))

    def show_state(self, state: dict[str, Any]) -> None:
        shown = (tuple(state["terrain_types"]), state["terrain_type"], tuple(state["map_layouts"]), state["map_layout"])
        if shown == self.shown:
            return
        self.shown = shown
        types, terrain, layouts, layout = shown
        for box, items, current in ((self.terrain, types, terrain), (self.layout_box, (NO_LAYOUT, *layouts), layout or NO_LAYOUT)):
            box.blockSignals(True)
            box.clear()
            box.addItems(list(items))
            box.setCurrentIndex(max(0, box.findText(current)) if current is not None else -1)
            box.blockSignals(False)
        self.terrain.setEnabled(bool(types))
        self.layout_box.setEnabled(bool(types))


class DebugView(QLabel):
    def show_state(self, state: dict[str, Any]) -> None:
        self.setText(debug_text(state))


class _ActionList(QListWidget):
    """A list that never takes the focus. It fills again only when its data
    changes, so that a click and the scroll position stay."""

    def __init__(self, dispatch: Callable[..., Any], parent: QWidget | None = None):
        super().__init__(parent)
        self.dispatch = dispatch
        self.shown: Any = None
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.itemClicked.connect(lambda item: self.on_row(self.row(item)))

    def fill(self, shown: Any, rows: list[tuple[str, str | None]]) -> None:
        if shown == self.shown:
            return
        self.shown = shown
        self.clear()
        for text, level in rows:
            item = QListWidgetItem(text)
            if level is not None:
                item.setForeground(QColor(COLORS[level]))
            self.addItem(item)

    def on_row(self, row: int) -> None:
        raise NotImplementedError


class OperativeList(_ActionList):
    """One row per base of the snapshot. A click on a row selects the base."""

    def show_state(self, state: dict[str, Any]) -> None:
        naming = state["naming"]
        self.fill((naming, state["bases"]), [self._row(base, naming) for base in state["bases"]])

    @staticmethod
    def _row(base: dict[str, Any], naming: bool) -> tuple[str, str | None]:
        size = f"{base['diameter_mm']:g} mm"
        if not naming:
            return size, None
        if base["label"] is None:
            return f"?   {size}", "error"
        name = base["label"].rsplit("/", 1)[-1]
        mark = "  (confirmed)" if base["confirmed"] else "" if base["score"] is None else f"   {percent(base['score'])}"
        return f"{name}{mark}", "ok" if base["sure"] or base["confirmed"] else "warn"

    def on_row(self, row: int) -> None:
        self.dispatch("base.select", index=self.shown[1][row]["index"])


class CandidateList(_ActionList):
    """The names for the selected base. A click on a row sets the name."""

    def show_state(self, state: dict[str, Any]) -> None:
        self.fill((state["selected"], state["candidates"]), [(candidate_text(c), None) for c in state["candidates"]])

    def on_row(self, row: int) -> None:
        index, candidates = self.shown
        self.dispatch("base.choose", label=candidates[row]["label"], index=index)


class VideoWidget(QWidget):
    """Shows a BGR image, scaled to fit with its aspect ratio. Reports clicks
    in image pixels and its width in screen pixels."""

    clicked = Signal(float, float, QPoint)  # image x, image y, global position of the click
    hovered = Signal(object)  # (image x, image y) of the mouse pointer, None when it is not on the image
    right_clicked = Signal(float, float, QPoint)
    width_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setMinimumSize(320, 240)
        self.setMouseTracking(True)  # pointer positions with no button down
        self._array: np.ndarray | None = None  # keeps the pixels of `_image` alive
        self._image: QImage | None = None
        self.badge: tuple[str, str, str] | None = None  # text, sign colour and sign at the top right

    def set_badge(self, text: str | None, color: str = "#ffffff", sign: str = "dot") -> None:
        """A small label at the top-right corner of the image, for example the
        mode. `sign`: `dot` (circle), `pause` (two bars) or `square`."""
        badge = None if text is None else (text, color, sign)
        if badge != self.badge:
            self.badge = badge
            self.update()

    def set_image(self, array: np.ndarray | None) -> None:
        if array is None or array is self._array:
            return
        self._array = array
        h, w = array.shape[:2]
        self._image = QImage(array.data, w, h, int(array.strides[0]), QImage.Format.Format_BGR888)
        self.update()

    def image_rect(self) -> QRectF | None:
        """Where the image is in the widget."""
        if self._image is None:
            return None
        scale = min(self.width() / self._image.width(), self.height() / self._image.height())
        w, h = self._image.width() * scale, self._image.height() * scale
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def to_image(self, pos: QPointF) -> tuple[float, float] | None:
        """Widget position to image pixels, or None outside the image."""
        rect = self.image_rect()
        if rect is None or not rect.contains(pos):
            return None
        scale = self._image.width() / rect.width()
        return (pos.x() - rect.x()) * scale, (pos.y() - rect.y()) * scale

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        rect = self.image_rect()
        if rect is not None:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawImage(rect, self._image)
        if self.badge is not None:
            self._paint_badge(painter, rect or QRectF(self.rect()))

    def _paint_badge(self, painter: QPainter, area: QRectF) -> None:
        text, color, sign = self.badge
        font = QFont(painter.font())
        font.setPointSizeF(max(10.0, font.pointSizeF() * 1.2))
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        dot, pad = metrics.height() * 0.6, metrics.height() * 0.4
        w, h = dot + 3 * pad + metrics.horizontalAdvance(text), metrics.height() + pad
        box = QRectF(area.right() - w - pad, area.top() + pad, w, h)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 170))
        painter.drawRoundedRect(box, h / 2, h / 2)
        painter.setBrush(QColor(color))
        mark = QRectF(box.left() + pad, box.center().y() - dot / 2, dot, dot)
        if sign == "pause":
            bar = dot * 0.36
            painter.drawRect(QRectF(mark.left(), mark.top(), bar, dot))
            painter.drawRect(QRectF(mark.right() - bar, mark.top(), bar, dot))
        elif sign == "square":
            painter.drawRect(mark)
        else:
            painter.drawEllipse(mark)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(QRectF(box.left() + dot + 2 * pad, box.top(), w - dot - 2 * pad, h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        signals = {Qt.MouseButton.LeftButton: self.clicked, Qt.MouseButton.RightButton: self.right_clicked}
        signal = signals.get(event.button())
        at = self.to_image(event.position())
        if signal is not None and at is not None:
            signal.emit(at[0], at[1], event.globalPosition().toPoint())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self.hovered.emit(self.to_image(event.position()))

    def leaveEvent(self, _event) -> None:
        self.hovered.emit(None)

    def resizeEvent(self, _event) -> None:
        self.width_changed.emit(int(self.width() * self.devicePixelRatioF()))


class RingSlider(QWidget):
    """A slider from 0 (no ring) to 9 inches for the selected base. A move of
    the user sends `ring.N` or `ring.clear`; the state moves it back."""

    def __init__(self, dispatch: Callable[..., Any], parent: QWidget | None = None):
        super().__init__(parent)
        self.dispatch = dispatch
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Distance"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 9)
        self.slider.setPageStep(1)
        self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.slider.setToolTip("Ring around the base. Keys: 1 to 9, 0 removes it")
        row.addWidget(self.slider, 1)
        self.value = QLabel()
        self.value.setMinimumWidth(32)
        row.addWidget(self.value)
        self.slider.valueChanged.connect(self._moved)
        self._show(0)

    @staticmethod
    def text(inches: int) -> str:
        return "Off" if inches == 0 else f'{inches}"'

    def _show(self, inches: int) -> None:
        self.value.setText(self.text(inches))

    def _moved(self, inches: int) -> None:
        self._show(inches)
        self.dispatch("ring.clear" if inches == 0 else f"ring.{inches}")

    def show_state(self, state: dict[str, Any]) -> None:
        inches = state["ring_in"] or 0
        self.setEnabled(state["selected"] is not None)
        if self.slider.value() != inches and not self.slider.isSliderDown():
            self.slider.blockSignals(True)
            self.slider.setValue(inches)
            self.slider.blockSignals(False)
            self._show(inches)


NO_TEAM = "-"
EYE = "\U0001F441"  # identified and confirmed


IDENT_ROLE = Qt.ItemDataRole.UserRole + 1  # identification mark in front of the name in a team list row
WOUNDS_ROLE = Qt.ItemDataRole.UserRole + 2  # wounds at the right edge of a team list row
SELECTED_ROLE = Qt.ItemDataRole.UserRole + 3  # True: the operative of the row is selected
SELECTED_BACK = QColor(128, 128, 128, 90)  # background of the selected row of a team list
MARK_PAD = 5  # px each side of the identification mark in a team list row
ROSTER_FONT_SCALE = 1.3  # team choice and operative list of a team side


def wounds_text(current: int | None, maximum: int | None) -> str:
    """`current/maximum` wounds, "" without a maximum."""
    return "" if maximum is None else f"{current}/{maximum}"


def roster_row(operative: dict[str, Any]) -> tuple[str, str, str, str]:
    """Name, identification mark (eye, percentage or `?`), wounds and level
    of one operative of a team list."""
    wounds = wounds_text(operative.get("wounds"), operative.get("max_wounds"))
    if operative["base_index"] is None:
        mark, level = "?", "none"
    elif operative["confirmed"]:
        mark, level = EYE, "ok"
    else:
        score = operative["score"]
        mark, level = "" if score is None else percent(score), "warn"
    return operative["name"], mark, wounds, level


def _larger_font(widget: QWidget) -> None:
    font = widget.font()
    font.setPointSizeF(font.pointSizeF() * ROSTER_FONT_SCALE)
    widget.setFont(font)


class _MarkDelegate(QStyledItemDelegate):
    """Draws a team list row: the identification mark in a column in front
    of the name, the name, and the wounds at the right edge, all in the
    colour of the row. The mark column is as wide as the widest mark of the
    list (`marks`), with `MARK_PAD` each side."""

    marks: tuple[str, ...] = ()

    def mark_width(self, metrics) -> int:
        widest = max((metrics.horizontalAdvance(m) for m in self.marks if m), default=0)
        return widest + 2 * MARK_PAD if widest else MARK_PAD

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        name = opt.text
        opt.text = ""
        widget = opt.widget
        if index.data(SELECTED_ROLE):
            painter.fillRect(opt.rect, SELECTED_BACK)
        (widget.style() if widget is not None else QApplication.style()).drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, widget)
        painter.save()
        painter.setFont(opt.font)
        brush = index.data(Qt.ItemDataRole.ForegroundRole)
        if brush is not None:
            painter.setPen(brush.color())
        rect = opt.rect
        mark_width = self.mark_width(opt.fontMetrics)
        middle = Qt.AlignmentFlag.AlignVCenter
        painter.drawText(rect.adjusted(MARK_PAD, 0, mark_width - MARK_PAD - rect.width(), 0), Qt.AlignmentFlag.AlignHCenter | middle, index.data(IDENT_ROLE) or "")
        wounds = index.data(WOUNDS_ROLE) or ""
        painter.drawText(rect.adjusted(0, 0, -6, 0), Qt.AlignmentFlag.AlignRight | middle, wounds)
        name_rect = rect.adjusted(mark_width, 0, -(opt.fontMetrics.horizontalAdvance(wounds) + 12), 0)
        painter.drawText(name_rect, Qt.AlignmentFlag.AlignLeft | middle, opt.fontMetrics.elidedText(name, Qt.TextElideMode.ElideRight, name_rect.width()))
        painter.restore()


class _RosterList(_ActionList):
    """Operatives of one team. An identified operative is in the colour of
    its side (`color`), the others grey. The selected operative has a grey
    background. A click on any row selects that operative."""

    def __init__(self, dispatch: Callable[..., Any], color: str | None = None, parent: QWidget | None = None):
        super().__init__(dispatch, parent)
        self.color = color
        self.operatives: list[dict[str, Any]] = []
        self.delegate = _MarkDelegate(self)
        self.setItemDelegate(self.delegate)

    def fill_rows(self, operatives: list[dict[str, Any]], rows: list[tuple[str, str, str, str]], selected: str | None = None) -> None:
        shown = (operatives, selected)
        if shown == self.shown:
            return
        self.operatives = operatives
        self.delegate.marks = tuple(mark for _name, mark, _wounds, _level in rows)
        self.fill(shown, [(name, level) for name, _mark, _wounds, level in rows])
        for k, (op, (_name, mark, wounds, _level)) in enumerate(zip(operatives, rows)):
            item = self.item(k)
            item.setData(IDENT_ROLE, mark)
            item.setData(WOUNDS_ROLE, wounds)
            item.setData(SELECTED_ROLE, selected is not None and op.get("label") == selected)
            if self.color is not None and op["base_index"] is not None:
                item.setForeground(QColor(self.color))

    def is_selected(self, row: int) -> bool:
        return bool(self.item(row).data(SELECTED_ROLE))

    def mark(self, row: int) -> str:
        """Identification mark of a row: eye, percentage or `?`."""
        return self.item(row).data(IDENT_ROLE)

    def wounds(self, row: int) -> str:
        return self.item(row).data(WOUNDS_ROLE)

    def on_row(self, row: int) -> None:
        self.dispatch("operative.select", label=self.operatives[row]["label"])


ADD_OPERATIVE = "Add operative"
ROSTER_BUTTON = "QPushButton { padding: 3px 4px; }"
CHOOSER_ROWS = 6  # most rows of the "Add operative" list without a scroll bar


class TeamRoster(QWidget):
    """One side of the board: a team choice and the list of its operatives.
    A click on an operative selects it (its base, when it has one).

    Below the list: "Add operative" opens a list of the team's operatives
    that are not in the team list (the button is then "Cancel"); a click on
    a name puts it at the end of the team list and closes the chooser.
    "Remove operative" takes the selected operative out of the list."""

    def __init__(self, slot: int, dispatch: Callable[..., Any], parent: QWidget | None = None):
        super().__init__(parent)
        self.slot = slot
        self.dispatch = dispatch
        self.shown_choices: Any = None
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(3)
        self.team = QComboBox()
        self.team.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.team.view().setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.team.setToolTip("Team on this side of the board")
        _larger_font(self.team)
        self.team.activated.connect(lambda i: self.dispatch("team.set", slot=self.slot, name=None if i == 0 else self.team.itemText(i)))
        column.addWidget(self.team)
        self.operatives = _RosterList(dispatch, team_color(slot))
        _larger_font(self.operatives)
        column.addWidget(self.operatives, 1)
        self.chooser = QListWidget()  # operatives that are not in the team list
        self.chooser.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.chooser.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.chooser.setToolTip("Click an operative to add it to the team list")
        _larger_font(self.chooser)
        self.chooser.itemClicked.connect(self._choose)
        self.chooser.setVisible(False)
        column.addWidget(self.chooser, 0)
        self.missing: list[dict[str, Any]] = []
        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(2)
        self.add = QPushButton(ADD_OPERATIVE)
        self.add.setToolTip("Add an operative of this team that is not in the list")
        self.add.clicked.connect(self._toggle_chooser)
        self.remove = QPushButton("Remove operative")
        self.remove.setToolTip("Remove the selected operative from the list: identification does not use it")
        self.remove.clicked.connect(lambda: self.dispatch("roster.remove", slot=self.slot))
        for button in (self.add, self.remove):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setAttribute(Qt.WidgetAttribute.WA_LayoutUsesWidgetRect)  # no extra macOS layout margin around the button
            button.setStyleSheet(ROSTER_BUTTON)  # small side padding: both labels fit in the team column
            buttons.addWidget(button)
        column.addLayout(buttons)

    def _toggle_chooser(self) -> None:
        self._show_chooser(not self.chooser.isVisible())

    def _show_chooser(self, on: bool) -> None:
        self.chooser.setVisible(on)
        self.add.setText("Cancel" if on else ADD_OPERATIVE)

    def _choose(self, item: QListWidgetItem) -> None:
        label = self.missing[self.chooser.row(item)]["label"]
        self._show_chooser(False)
        self.dispatch("roster.add", slot=self.slot, label=label)

    def show_state(self, state: dict[str, Any]) -> None:
        slots = state["team_slots"]
        other = slots[1 - self.slot] if len(slots) == 2 else None
        choices = (tuple(state["team_choices"]), slots[self.slot], other)
        if choices != self.shown_choices:
            self.shown_choices = choices
            items, current, other = choices
            self.team.blockSignals(True)
            self.team.clear()
            self.team.addItems([NO_TEAM, *items])
            model = self.team.model()
            for k, name in enumerate(items, start=1):  # the team of the other side is not a choice
                model.item(k).setEnabled(name != other)
            self.team.setCurrentIndex(max(0, self.team.findText(current)) if current is not None else 0)
            self.team.blockSignals(False)
        roster = state["rosters"][self.slot]
        rows = [] if roster is None else roster["operatives"]
        self.operatives.fill_rows(rows, [roster_row(r) for r in rows], state.get("selected_label"))
        missing = [] if roster is None else roster["missing"]
        if missing != self.missing:
            self.missing = missing
            self.chooser.clear()
            self.chooser.addItems([m["name"] for m in missing])
            rows_shown = min(len(missing), CHOOSER_ROWS)  # as high as its names, up to CHOOSER_ROWS
            self.chooser.setFixedHeight(rows_shown * max(self.chooser.sizeHintForRow(0), 1) + 2 * self.chooser.frameWidth())
        if not missing and self.chooser.isVisible():
            self._show_chooser(False)
        self.add.setEnabled(bool(missing) or self.chooser.isVisible())
        self.remove.setEnabled(any(r["label"] == state.get("selected_label") for r in rows))


class WoundsControl(QWidget):
    """Current wounds of the selected operative: a - and a + button and a
    box to type a number from 0 to the maximum. A warning when the operative
    has no maximum wounds. Hidden when no named operative is selected.

    The box takes the focus only for a click, so the keys work again after
    Return or a click elsewhere."""

    def __init__(self, dispatch: Callable[..., Any], parent: QWidget | None = None):
        super().__init__(parent)
        self.dispatch = dispatch
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)
        self.controls = QWidget()
        row = QHBoxLayout(self.controls)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("Wounds"))
        actions = {a.id: a for a in ACTIONS}
        self.minus = action_control(actions["wounds.minus"], dispatch)
        self.minus.setText("-")
        self.plus = action_control(actions["wounds.plus"], dispatch)
        self.plus.setText("+")
        self.value = QSpinBox()
        self.value.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
        self.value.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.value.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.value.setKeyboardTracking(False)
        self.value.setToolTip("Type the current wounds, then push Return")
        self.value.editingFinished.connect(self._typed)
        self.maximum = QLabel()
        for widget in (self.minus, self.value, self.maximum, self.plus):
            row.addWidget(widget)
        row.addStretch(1)
        column.addWidget(self.controls)
        self.warning = QLabel()
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet(f"color: {COLORS['warn']}")
        column.addWidget(self.warning)
        self.shown: dict[str, Any] | None = None

    def _typed(self) -> None:
        self.value.clearFocus()
        w = self.shown
        if w is not None and w["max"] is not None and self.value.value() != w["current"]:
            self.dispatch("wounds.set", value=self.value.value())

    def show_state(self, state: dict[str, Any]) -> None:
        w = state["selected_wounds"]
        self.shown = w
        self.setVisible(w is not None)
        if w is None:
            return
        has_max = w["max"] is not None
        self.controls.setVisible(has_max)
        self.warning.setVisible(not has_max)
        if not has_max:
            name = w["label"].rsplit("/", 1)[-1]
            self.warning.setText(f"{name} has no maximum wounds set.")
            return
        self.minus.setEnabled(w["current"] > 0)
        self.plus.setEnabled(w["current"] < w["max"])
        self.maximum.setText(f"/ {w['max']}")
        if not self.value.hasFocus():  # the user types a number
            self.value.blockSignals(True)
            self.value.setRange(0, w["max"])
            self.value.setValue(w["current"])
            self.value.blockSignals(False)


OPEN_STATE = {False: "Closed", True: "Open"}


class OpeningControl(QWidget):
    """State of the selected hatchway or breach point and a button that
    opens or closes it."""

    def __init__(self, dispatch: Callable[..., Any], parent: QWidget | None = None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.state_label = QLabel()
        row.addWidget(self.state_label)
        self.button = action_control(next(a for a in ACTIONS if a.id == "opening.toggle"), dispatch)
        row.addWidget(self.button)
        row.addStretch(1)

    def show_state(self, state: dict[str, Any]) -> None:
        opening = state["selected_opening"]
        self.setEnabled(opening is not None)
        is_open = opening is not None and opening["open"]
        self.state_label.setText(OPEN_STATE[is_open])
        self.button.setText("Close" if is_open else "Open")


NO_TAC_OP = "(no Tac Op)"
PRIMARY_NAMES = {"hidden": "Hidden", "crit": "Crit Op", "tac": "Tac Op", "kill": "Kill Op"}
SHORT_NAMES = {"crit": "Crit", "tac": "Tac", "kill": "Kill"}
SKULL_ON = "#e8c547"
SKULL_OFF = "#5a5a5a"
SKULL_PX = 16  # icon size of one skull
SKULL_BUTTON = "QToolButton { border: none; padding: 1px; }"


def skull_path(size: float) -> QPainterPath:
    """Outline of a skull in a `size` square: round cranium, narrower jaw
    with teeth gaps, two eye holes and a nose hole cut out."""
    s = size
    head = QPainterPath()
    head.addEllipse(QRectF(0.08 * s, 0.02 * s, 0.84 * s, 0.70 * s))
    jaw = QPainterPath()
    jaw.addRoundedRect(QRectF(0.26 * s, 0.50 * s, 0.48 * s, 0.46 * s), 0.08 * s, 0.08 * s)
    shape = head.united(jaw)
    holes = QPainterPath()
    holes.addEllipse(QRectF(0.22 * s, 0.32 * s, 0.22 * s, 0.22 * s))
    holes.addEllipse(QRectF(0.56 * s, 0.32 * s, 0.22 * s, 0.22 * s))
    nose = QPainterPath(QPointF(0.50 * s, 0.56 * s))
    nose.lineTo(0.44 * s, 0.68 * s)
    nose.lineTo(0.56 * s, 0.68 * s)
    nose.closeSubpath()
    holes.addPath(nose)
    for x in (0.40, 0.50, 0.60):  # gaps between the teeth
        holes.addRect(QRectF((x - 0.015) * s, 0.80 * s, 0.03 * s, 0.17 * s))
    return shape.subtracted(holes)


_skull_icons: dict[str, QIcon] = {}


def skull_icon(color: str) -> QIcon:
    """A filled skull icon of one colour, sharp on high-DPI screens."""
    if color not in _skull_icons:
        ratio = 3
        pixmap = QPixmap(SKULL_PX * ratio, SKULL_PX * ratio)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillPath(skull_path(SKULL_PX * ratio), QColor(color))
        painter.end()
        pixmap.setDevicePixelRatio(ratio)
        _skull_icons[color] = QIcon(pixmap)
    return _skull_icons[color]
GRID_LINE = "1px solid #8a8a8a"


def cell_border(category: str, tp: int | None) -> str:
    """Style of a Crit or Tac cell: lines between the cells and around the
    grid, each line drawn once. Kill has no grid."""
    if tp is None:
        return ""
    sides = ["right", "bottom"]
    if category == CATEGORIES[0]:
        sides.append("top")
    if tp == TURNING_POINTS[0]:
        sides.append("left")
    return "#cell { " + " ".join(f"border-{s}: {GRID_LINE};" for s in sides) + " }"


def total_text(score: dict[str, Any]) -> str:
    """`14 VP`."""
    return f"{score['total']} VP"


def team_color(slot: int) -> str:
    """Qt colour (`#rrggbb`) of the team circles of one side (BGR in `live.TEAM_COLORS`)."""
    b, g, r = TEAM_COLORS[slot]
    return f"#{r:02x}{g:02x}{b:02x}"


def _no_focus_combo(box: QComboBox) -> QComboBox:
    box.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    box.view().setFocusPolicy(Qt.FocusPolicy.NoFocus)
    return box


class ScoreCounter(QGroupBox):
    """Score of one side of the board: command points with - and + on top,
    Tac Op and Primary choices (the Primary bonus at the end of its row), a skull
    grid (Crit and Tac per turning point 2 to 4, Kill in one cell), command
    total in a big font, in the colour of the side's team. A click on a skull fills
    its cell up to it; a click on the last lit skull turns it off."""

    def __init__(self, slot: int, dispatch: Callable[..., Any], parent: QWidget | None = None):
        super().__init__(parent)
        self.slot = slot
        self.dispatch = dispatch
        self.shown_tac_ops: list[str] | None = None
        column = QVBoxLayout(self)
        column.setContentsMargins(6, 4, 6, 6)
        column.setSpacing(4)

        choices = QFormLayout()
        choices.setContentsMargins(0, 0, 0, 0)
        choices.setVerticalSpacing(2)
        self.tac_op = _no_focus_combo(QComboBox())
        self.tac_op.setToolTip("Tac Op of this side, from the archetypes of the team")
        self.tac_op.activated.connect(lambda i: self.dispatch("score.tac_op", slot=self.slot, name=None if i == 0 else self.tac_op.itemText(i)))
        choices.addRow("Tac Op", self.tac_op)
        self.primary = _no_focus_combo(QComboBox())
        self.primary.addItems([PRIMARY_NAMES[p] for p in PRIMARIES])
        self.primary.setToolTip("Primary op: adds half of its VP, rounded up. Hidden adds nothing.")
        self.primary.activated.connect(lambda i: self.dispatch("score.primary", slot=self.slot, value=PRIMARIES[i]))
        self.bonus = QLabel()
        self.bonus.setMinimumWidth(24)
        self.bonus.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.bonus.setToolTip("Primary points: half of the VP of the Primary op, rounded up")
        primary_row = QHBoxLayout()
        primary_row.setContentsMargins(0, 0, 0, 0)
        primary_row.addWidget(self.primary, 1)
        primary_row.addWidget(self.bonus, 0)
        choices.addRow("Primary", primary_row)
        column.addLayout(choices)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(0)
        grid.setVerticalSpacing(0)
        for k, tp in enumerate(TURNING_POINTS, start=1):
            grid.addWidget(QLabel(f"TP{tp}"), 0, k, Qt.AlignmentFlag.AlignHCenter)
        vp = QLabel("VP")
        vp.setContentsMargins(4, 0, 0, 0)
        grid.addWidget(vp, 0, len(TURNING_POINTS) + 1, Qt.AlignmentFlag.AlignRight)
        self.skulls: dict[tuple[str, int | None], list[QToolButton]] = {}
        self.points: dict[str, QLabel] = {}
        for row, category in enumerate(CATEGORIES, start=1):
            name = QLabel(SHORT_NAMES[category])
            name.setToolTip(CATEGORY_NAMES[category])
            name.setContentsMargins(0, 0, 4, 0)
            grid.addWidget(name, row, 0)
            if category == "kill":
                grid.addWidget(self._cell(category, None, CATEGORY_MAX), row, 1, 1, len(TURNING_POINTS), Qt.AlignmentFlag.AlignHCenter)
            else:
                for k, tp in enumerate(TURNING_POINTS, start=1):
                    grid.addWidget(self._cell(category, tp, SKULLS_PER_TP), row, k)
            self.points[category] = QLabel("0")
            grid.addWidget(self.points[category], row, len(TURNING_POINTS) + 1, Qt.AlignmentFlag.AlignRight)
        column.addLayout(grid)

        cp_row = QHBoxLayout()
        cp_row.setContentsMargins(0, 0, 0, 0)
        self.cp_minus = QPushButton("-")
        self.cp_plus = QPushButton("+")
        self.cp = QLabel()
        self.cp.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = self.cp.font()
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() * 1.6)
        self.cp.setFont(font)
        self.cp.setMinimumWidth(32)
        for button, delta in ((self.cp_minus, -1), (self.cp_plus, 1)):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setFixedWidth(32)
            button.clicked.connect(lambda _checked=False, d=delta: self.dispatch("score.cp", slot=self.slot, delta=d))
        cp_row.addStretch(1)
        for widget in (self.cp_minus, self.cp, self.cp_plus):
            cp_row.addWidget(widget)
        cp_row.addStretch(1)  # centred in the field column of Tac Op and Primary
        cp_label = QLabel("CP")
        cp_label.setToolTip("Command points")
        label_font = cp_label.font()
        label_font.setPointSizeF(font.pointSizeF())  # same size as the value
        cp_label.setFont(label_font)
        choices.insertRow(0, cp_label, cp_row)  # on top

        self.total = QLabel()
        font = self.total.font()
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() * 2.2)
        self.total.setFont(font)
        self.total.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.total.setToolTip("Crit + Tac + Kill + Primary")
        self.total.setStyleSheet(f"color: {team_color(slot)}")
        column.addWidget(self.total)

    def _cell(self, category: str, tp: int | None, count: int) -> QWidget:
        cell = QWidget()
        cell.setObjectName("cell")
        cell.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)  # draw the border
        cell.setStyleSheet(cell_border(category, tp))
        row = QHBoxLayout(cell)
        row.setContentsMargins(2, 0, 2, 0)
        row.setSpacing(0)
        row.addStretch(1)
        buttons = []
        for skull in range(1, count + 1):
            button = QToolButton()
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setStyleSheet(SKULL_BUTTON)
            button.setIconSize(QSize(SKULL_PX, SKULL_PX))
            button.setIcon(skull_icon(SKULL_OFF))
            button.setProperty("lit", False)
            button.clicked.connect(lambda _checked=False, s=skull: self.dispatch("score.skull", slot=self.slot, category=category, tp=tp, skull=s))
            row.addWidget(button)
            buttons.append(button)
        row.addStretch(1)
        self.skulls[(category, tp)] = buttons
        return cell

    def lit(self, category: str, tp: int | None) -> int:
        """Number of lit skulls of a cell (for tests)."""
        return sum(bool(b.property("lit")) for b in self.skulls[(category, tp)])

    def show_state(self, state: dict[str, Any]) -> None:
        score = state["scores"][self.slot]
        for (category, tp), buttons in self.skulls.items():
            value = score["cells"][category][0 if tp is None else TURNING_POINTS.index(tp)]
            for k, button in enumerate(buttons, start=1):
                lit = k <= value
                if bool(button.property("lit")) != lit:
                    button.setProperty("lit", lit)
                    button.setIcon(skull_icon(SKULL_ON if lit else SKULL_OFF))
        for category, label in self.points.items():
            label.setText(str(score["points"][category]))
        if score["tac_ops"] != self.shown_tac_ops:
            self.shown_tac_ops = list(score["tac_ops"])
            self.tac_op.blockSignals(True)
            self.tac_op.clear()
            self.tac_op.addItems([NO_TAC_OP, *self.shown_tac_ops])
            self.tac_op.blockSignals(False)
        self.tac_op.setEnabled(bool(score["tac_ops"]))
        self.tac_op.setCurrentIndex(max(0, self.tac_op.findText(score["tac_op"])) if score["tac_op"] is not None else 0)
        self.primary.setCurrentIndex(PRIMARIES.index(score["primary"]))
        self.cp.setText(str(score["cp"]))
        self.cp_minus.setEnabled(score["cp"] > 0)
        self.bonus.setText(f"+{score['primary_bonus']}")
        self.total.setText(total_text(score))
