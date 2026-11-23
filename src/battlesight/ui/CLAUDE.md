# src/battlesight/ui: Qt front end

PySide6, imported only inside `app.cmd_live`. The game core is `live.py` (`docs/dev/live-session.md`). `docs/specs/ui.md` is the UI plan: many layouts on the `LiveSession` contract. Tests run offscreen: `tests/test_ui.py` (run `uv run pytest -m ui`).

## Files

- `shell.py`: `Worker` (a `QThread` that owns the camera and the `LiveSession`; the GUI sends commands to its queue, they run between frames; old frames are dropped), `MainWindow` (one `QShortcut` per action, menu generated from `ACTIONS`, status bar (status text, notice, debug values, then `ProgressView` at the right end; the worker sends `progress_ready` at each step of a long action, a frame hides it), name popup, `set_layout`; `dispatch` drops commands while `state["busy"]`), `run_window`.
- `parts.py`: shared widgets: `ProgressView` ("<task text>:" and a `QProgressBar` of `state["progress"]`, hidden when `null`), `VideoWidget` (`NoFocus`, widget to image pixels; the main video is `zoomable=True`, crops are not: wheel zooms at the pointer, middle button drags, trackpad pinch and two-finger scroll, `-` `+` and a percentage at the top right (shown above 100%, a click resets), 100% to 800%; above 100% `view_changed` sends a `live.ZoomView` and the worker passes it to `set_display_view`; the frame is then only the visible region at the screen scale (`set_image(array, region)`), and the last whole image fills the edges until the new region comes; clicks stay in pixels of the whole image at 100%; a new layout keeps the view; `Ctrl+0` resets), action controls, `MapPicker`, `DebugView`, `OperativeList`, `CandidateList`, `WoundsControl` (its number box is the one control with `ClickFocus`: typing needs the focus, Return gives it back), `OpeningControl`, `LightControl`, `ScoreCounter`, `GameControl`, `ring_controls`, `aim_texts`.
- `layouts/`: `LAYOUTS`:
  - `rosters` = default. Top bar with View / Commands (Debug check box at the bottom) / Inspector / Reference sections and a `TeamRoster` each side. Game bar between top bar and video: `GameControl` in the centre (`TURN: - X +`, Crit Op list), team choice of each side (`TeamRoster(team_on_top=False)`, the layout places `roster.team`). `ScoreCounter` (no title; CP on top, total in the team colour) below each team list. Badge: Live = red dot, Snapshot = blue pause sign, Minimap = green square.
  - `inspector` = state-driven panel with the name list.
  - `panel` = controls generated from `ACTIONS`, name popup.
- No settings file: the layout name goes to the session (`view.layout`) and lives in the saved game state (`--state`). Layouts take no options.

## Rules

- A layout only places widgets and uses `shell.dispatch`, `shell.new_video`, `shell.selected_crop`, `shell.show_debug`. It must not import `pipeline`, `ident`, `calib` or `camera` (a test checks this).
- Every control and list needs `NoFocus`, or `space` pushes it.
- Status text and the name menu are drawn from `state`. `Action.enabled` and `checked` only tell the front end how to show a control. `shell = True` actions (help, quit) belong to the front end.
- Never call `cv2.imshow` or `cv2.waitKey`.
- Left and right click are for the board. The view uses the wheel and the middle button only.
- Right click: with `state["right_click"] = "name"` the shell opens the name popup in layouts with `name_popup`.
- While `shooting` the shell sends the pointer through `board.hover`.
