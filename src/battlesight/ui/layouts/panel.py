"""Layout `panel`: the video and one fixed panel with one control per action.

The panel comes from `ACTIONS`, so a new action needs no work here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from PySide6.QtWidgets import QAbstractButton, QGroupBox, QHBoxLayout, QVBoxLayout, QWidget

from ...live import ACTIONS, Action
from ..parts import DebugView, action_control, action_group, panel_side, ring_controls, show_action_state

if TYPE_CHECKING:
    from . import Shell


class PanelLayout:
    name: ClassVar[str] = "panel"
    name_popup: ClassVar[bool] = True

    def __init__(self) -> None:
        self.controls: dict[str, tuple[Action, QAbstractButton]] = {}

    def build(self, shell: Shell, options: dict[str, Any]) -> QWidget:
        side = panel_side(options)
        root = QWidget()
        row = QHBoxLayout(root)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(shell.new_video(), 1)
        row.insertWidget(0 if side == "left" else 1, self._panel(shell), 0)
        return root

    def _panel(self, shell: Shell) -> QWidget:
        panel = QWidget()
        column = QVBoxLayout(panel)
        groups: dict[str, QVBoxLayout] = {}
        for action in ACTIONS:
            if action.args or action.id.startswith("ring."):
                continue
            title = action_group(action)
            if title not in groups:
                box = QGroupBox(title)
                groups[title] = QVBoxLayout(box)
                column.addWidget(box)
            control = action_control(action, shell.dispatch)
            groups[title].addWidget(control)
            self.controls[action.id] = (action, control)
        box = QGroupBox("Ring")
        grid, rings = ring_controls(shell.dispatch)
        QVBoxLayout(box).addWidget(grid)
        column.addWidget(box)
        self.controls.update((action.id, (action, control)) for action, control in rings)
        box = QGroupBox("Debug")
        self.debug = DebugView()
        QVBoxLayout(box).addWidget(self.debug)
        column.addWidget(box)
        column.addStretch(1)
        return panel

    def on_state(self, state: dict[str, Any]) -> None:
        for action, control in self.controls.values():
            show_action_state(control, action, state)
        self.debug.show_state(state)
