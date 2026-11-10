"""Qt widgets that all layouts share."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QMouseEvent, QPainter
from PySide6.QtWidgets import QAbstractButton, QCheckBox, QStyledItemDelegate, QStyleOptionViewItem, QComboBox, QFormLayout, QGridLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QSlider, QVBoxLayout, QWidget

from ..live import ACTIONS, Action

ACTION_GROUPS = {"team": "View", "map": "View", "snapshot": "Snapshot", "base": "Snapshot", "names": "Snapshot", "frame": "Snapshot", "ring": "Measure", "view": "View", "reference": "Reference", "app": "Help"}
RING_COLUMNS = 5
COLORS = {"none": "#8a8a8a", "ok": "#2e9e4f", "done": "#2e9e4f", "warn": "#c98a00", "hint": "#c98a00", "error": "#d03a3a"}
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


def distance_text(distance: dict[str, float]) -> str:
    return f'{distance["in"]:.2f}"'


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
    right_clicked = Signal(float, float, QPoint)
    width_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setMinimumSize(320, 240)
        self._array: np.ndarray | None = None  # keeps the pixels of `_image` alive
        self._image: QImage | None = None
        self.badge: tuple[str, str] | None = None  # text and dot colour at the top right

    def set_badge(self, text: str | None, color: str = "#ffffff") -> None:
        """A small label at the top-right corner of the image, for example the mode."""
        badge = None if text is None else (text, color)
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
        text, color = self.badge
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
        painter.drawEllipse(QRectF(box.left() + pad, box.center().y() - dot / 2, dot, dot))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(QRectF(box.left() + dot + 2 * pad, box.top(), w - dot - 2 * pad, h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, text)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        signals = {Qt.MouseButton.LeftButton: self.clicked, Qt.MouseButton.RightButton: self.right_clicked}
        signal = signals.get(event.button())
        at = self.to_image(event.position())
        if signal is not None and at is not None:
            signal.emit(at[0], at[1], event.globalPosition().toPoint())

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


MARK_ROLE = Qt.ItemDataRole.UserRole + 1  # text at the right edge of a team list row


def roster_row(operative: dict[str, Any]) -> tuple[str, str, str]:
    """Name, mark (eye, percentage or `?`) and level of one operative of a team list."""
    if operative["base_index"] is None:
        return operative["name"], "?", "none"
    if operative["confirmed"]:
        return operative["name"], EYE, "ok"
    score = operative["score"]
    return operative["name"], "" if score is None else percent(score), "warn"


class _MarkDelegate(QStyledItemDelegate):
    """Draws the mark of a row at its right edge, in the colour of the row."""

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        super().paint(painter, option, index)
        mark = index.data(MARK_ROLE)
        if not mark:
            return
        painter.save()
        brush = index.data(Qt.ItemDataRole.ForegroundRole)
        if brush is not None:
            painter.setPen(brush.color())
        painter.drawText(option.rect.adjusted(0, 0, -6, 0), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, mark)
        painter.restore()


class _RosterList(_ActionList):
    def __init__(self, dispatch: Callable[..., Any], parent: QWidget | None = None):
        super().__init__(dispatch, parent)
        self.setItemDelegate(_MarkDelegate(self))

    def fill_rows(self, shown: Any, rows: list[tuple[str, str, str]]) -> None:
        if shown == self.shown:
            return
        self.fill(shown, [(name, level) for name, _mark, level in rows])
        for k, (_name, mark, _level) in enumerate(rows):
            self.item(k).setData(MARK_ROLE, mark)

    def mark(self, row: int) -> str:
        return self.item(row).data(MARK_ROLE)

    def on_row(self, row: int) -> None:
        index = self.shown[row]["base_index"]
        if index is not None:
            self.dispatch("base.select", index=index)


class TeamRoster(QWidget):
    """One side of the board: a team choice and the list of its operatives.
    A click on an identified operative selects its base."""

    def __init__(self, slot: int, dispatch: Callable[..., Any], parent: QWidget | None = None):
        super().__init__(parent)
        self.slot = slot
        self.dispatch = dispatch
        self.shown_choices: Any = None
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        self.team = QComboBox()
        self.team.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.team.view().setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.team.setToolTip("Team on this side of the board")
        self.team.activated.connect(lambda i: self.dispatch("team.set", slot=self.slot, name=None if i == 0 else self.team.itemText(i)))
        column.addWidget(self.team)
        self.operatives = _RosterList(dispatch)
        column.addWidget(self.operatives, 1)

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
        self.operatives.fill_rows(rows, [roster_row(r) for r in rows])
