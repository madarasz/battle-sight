"""Qt widgets that all layouts share."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter
from PySide6.QtWidgets import QAbstractButton, QCheckBox, QLabel, QListWidget, QListWidgetItem, QPushButton, QWidget

from ..live import Action

ACTION_GROUPS = {"snapshot": "Snapshot", "base": "Snapshot", "names": "Snapshot", "frame": "Snapshot", "view": "View", "reference": "Reference", "app": "Help"}
COLORS = {"ok": "#2e9e4f", "done": "#2e9e4f", "warn": "#c98a00", "hint": "#c98a00", "error": "#d03a3a"}
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


def candidate_text(candidate: dict[str, Any]) -> str:
    score = candidate["score"]
    return candidate["label"] if score is None else f"{candidate['label']}   {score:.2f}"


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


class StatusView(QLabel):
    """Drift status or camera state, in its colour."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWordWrap(True)

    def show_state(self, state: dict[str, Any]) -> None:
        drift = state["drift"]
        if not state["camera"]:
            text, level = "camera down", "error"
        elif drift is None:
            text, level = "waiting for the first frame", "warn"
        else:
            text, level = drift["text"], DRIFT_LEVEL[drift["status"]]
        self.setText(text)
        self.setStyleSheet(f"color: {COLORS[level]}")


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
        mark = "  (confirmed)" if base["confirmed"] else "" if base["score"] is None else f"   {base['score']:.2f}"
        return f"{name}{mark}", "ok" if base["sure"] or base["confirmed"] else "warn"

    def on_row(self, row: int) -> None:
        x, y = self.shown[1][row]["position_px"]
        self.dispatch("board.click", x=x, y=y)


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
    width_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setMinimumSize(320, 240)
        self._array: np.ndarray | None = None  # keeps the pixels of `_image` alive
        self._image: QImage | None = None

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

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() is not Qt.MouseButton.LeftButton:
            return
        at = self.to_image(event.position())
        if at is not None:
            self.clicked.emit(at[0], at[1], event.globalPosition().toPoint())

    def resizeEvent(self, _event) -> None:
        self.width_changed.emit(int(self.width() * self.devicePixelRatioF()))
