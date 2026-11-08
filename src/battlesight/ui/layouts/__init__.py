"""The layouts of the window. A layout only places widgets: it has no game
logic and no game state. Add a layout here, in the plain dict `LAYOUTS`.

A layout must not import `pipeline`, `ident`, `calib` or `camera`. A test checks this.
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

import numpy as np
from PySide6.QtWidgets import QWidget

from ..parts import VideoWidget
from .inspector import InspectorLayout
from .panel import PanelLayout


class Shell(Protocol):
    """What a layout can use of the window."""

    def dispatch(self, action_id: str, **args) -> int | None:
        """Do one action. Returns its command number, or None when it was not sent."""

    def new_video(self) -> VideoWidget:
        """A video widget that is connected to the window. Call it once in `build`."""

    def selected_crop(self) -> np.ndarray | None:
        """The board around the selected base, of the frame that `on_state` shows."""


class Layout(Protocol):
    name: ClassVar[str]
    name_popup: ClassVar[bool]  # the shell opens its name popup at a clicked base

    def build(self, shell: Shell, options: dict[str, Any]) -> QWidget:
        """Make and place the parts. A bad value in `options` is a ValueError."""

    def on_state(self, state: dict[str, Any]) -> None:
        """Show the new state."""


LAYOUTS: dict[str, type[Layout]] = {InspectorLayout.name: InspectorLayout, PanelLayout.name: PanelLayout}
DEFAULT_LAYOUT = InspectorLayout.name
