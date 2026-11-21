# src/battlesight/ui: Qt front end

PySide6, imported only inside `app.cmd_live`. The game core is `live.py` (`docs/dev/live-session.md`). `docs/specs/ui.md` is the UI plan: many layouts on the `LiveSession` contract. Tests run offscreen: `tests/test_ui.py` (run `uv run pytest -m ui`).

## Files

- `shell.py`: `Worker` (a `QThread` that owns the camera and the `LiveSession`; the GUI sends commands to its queue, they run between frames; old frames are dropped), `MainWindow` (one `QShortcut` per action, menu generated from `ACTIONS`, status bar, name popup, `set_layout`; `dispatch` drops commands while `state["busy"]`), `run_window`.
- `parts.py`: shared widgets: `VideoWidget` (`NoFocus`, widget to image pixels), action controls, `MapPicker`, `DebugView`, `OperativeList`, `CandidateList`, `WoundsControl` (its number box is the one control with `ClickFocus`: typing needs the focus, Return gives it back), `OpeningControl`, `LightControl`, `ScoreCounter`, `GameControl`, `ring_controls`, `aim_texts`.
- `layouts/`: `LAYOUTS`:
  - `rosters` = default. Top bar with View (its last check box is Debug) / Commands / Inspector / Reference sections and a `TeamRoster` each side. Game bar between top bar and video: `GameControl` in the centre (`TURN: - X +`, Crit Op list), team choice of each side (`TeamRoster(team_on_top=False)`, the layout places `roster.team`). `ScoreCounter` (no title; CP on top, total in the team colour) below each team list. Badge: Live = red dot, Snapshot = blue pause sign, Minimap = green square.
  - `inspector` = state-driven panel with the name list.
  - `panel` = controls generated from `ACTIONS`, name popup.
- No settings file: the layout name goes to the session (`view.layout`) and lives in the saved game state (`--state`). Layouts take no options.

## Rules

- A layout only places widgets and uses `shell.dispatch`, `shell.new_video`, `shell.selected_crop`, `shell.show_debug`. It must not import `pipeline`, `ident`, `calib` or `camera` (a test checks this).
- Every control and list needs `NoFocus`, or `space` pushes it.
- Status text and the name menu are drawn from `state`. `Action.enabled` and `checked` only tell the front end how to show a control. `shell = True` actions (help, quit) belong to the front end.
- Never call `cv2.imshow` or `cv2.waitKey`.
- Right click: with `state["right_click"] = "name"` the shell opens the name popup in layouts with `name_popup`.
- While `shooting` the shell sends the pointer through `board.hover`.
