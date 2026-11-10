# Battle Sight: UI plan

Status: all steps of section 12 are done. The layouts are `inspector` (default) and `panel`. The UI is modular: many layouts on one game core (section 8). To continue the work, read section 13 first. Date: 2026-11-08.

## 1. Goal

Replace the OpenCV window of the live app with a real UI. The live app is `battlesight` with no subcommand (`docs/cli.md`, "`battlesight`: the live app").

The UI must:

- Give each function of the live app a visible control. The keys stay.
- Move status text and messages off the video.
- Have a good place for the features of priorities 5 to 7 and for wounds and statuses (section 4).
- Have more than one UI layout. Each layout maps to the same controls and the same game state. A setting changes the layout (section 8).
- Keep the core free of Qt, so that a FastAPI and browser front end can use it later (`specs/plan.md`, section 6).

Not in scope: `enroll`, `gallery`, `calibrate`, `image-pipeline` and the tools. They stay as they are.

## 2. What changed since the first version of this file

| First version | Now | Result for the plan |
|---|---|---|
| `corners` and `rectify` are two live commands. | One live app, `cmd_live`, with no subcommand. | Do not add `battlesight ui`. The UI replaces the `imshow` loop of `cmd_live`. |
| `r` key and a re-lock button. | `DriftMonitor` locks and locks again by itself. No key. | Remove the re-lock button. |
| Only a live view with toggles. | Two states: live view and snapshot. A snapshot holds base results, names and fixes (`_Live`). | The UI structure must show these states. |
| No clicks. | Click a base, then click a name (`overlay.Menu`, `LabelSession`). | The menu can stay on the frame or become a widget. The options differ here. |
| Status banner only. | Banner, hint strip, `_Notice` strip (6 s), live drift strip over a snapshot, key list (`h`). | All of these move to widgets. |
| Fast loop only. | `space` runs DINOv2. The first `space` also loads the model and the gallery, which can take some seconds. | The GUI must not stop during this work (section 7). |
| New options were possible. | Option rule in `docs/cli.md`: no option for a thing that a key or a button can change. | No new options. No `--no-ui`. |

## 3. Functions of the live app today

| Function | Key or mouse | Available in | State to show |
|---|---|---|---|
| Find bases and names | `space` | live, snapshot | busy, result count |
| Find bases only (key and menu, no button) | `b` | live, snapshot | busy, result count |
| "Adjust perspective": top-down (default) or camera view | `v` | live, snapshot | which view |
| Grid | `g` | live, snapshot | on or off |
| Take empty-board reference | `e` | live, snapshot | reference is set or not set |
| Clear reference | `c` | live, snapshot | reference is set or not set |
| Select a base, measure to a second base | click a base, click another base | snapshot | selected base, target, distance |
| Ring around the selected base | `1` to `9`, `0` removes | snapshot, base selected | ring size |
| Fix a name | right click a base, click a name | snapshot with names | selected base, candidates, scores |
| Accept all names | `a` | snapshot with names | number not confirmed |
| Undo last gallery photo | `u` | snapshot with names | undo is possible or not |
| Save camera frame | `s` | live, snapshot | message |
| Key list | `h` | live, snapshot | replaced by tooltips, keep a help dialog |
| Clear selection, close snapshot | `Esc` | snapshot | back to live |
| Quit | `q` | always | |

Read-only status: drift status (the five `DriftStatus` values, and "board position updated", which is `OK` with `relocked`), lost tags or moved card, lens source, teams in use, messages of keys and clicks.

Values that the app does not show today, for the debug area: tags found (`DriftReport.visible`), fit error (`BoardView.reproj_max_mm`), fps and detect time (new, simple timers). A tag overlay toggle (from the first version) needs one new flag in the live drawing.

## 4. Features that the structure must have space for

| Source | Feature | UI need |
|---|---|---|
| Priority 5 | Select an operative, ring radius 1 to 9 in, second operative gives the rim-to-rim distance | Selection, keys `1` to `9`, a place for the distance |
| Priority 6 | Terrain layout of the game, from a map file (done) | Terrain type and layout lists, toggles for layout, drop zones and terrain keys |
| Priority 7 | Tokens, rings, list of operatives in range | Selection of a token, a result list |
| `specs/about-project`, plan section 9 | Wounds and statuses per operative | A persistent list of operatives |

## 5. Structure options

Each option is a possible layout (section 8). The app can have more than one. The comparison below helps to select which layouts to build and in which order.

All sizes are for a 1920 x 1080 window. The top-down image is about 1.16 : 1, so at full height it is about 1250 px wide and about 670 px of the width are free. The camera view is 16:9, so each panel makes it smaller.

### Option 1: fixed control panel

The direct upgrade of the first version. One panel, always the same, with one control per function.

```
+--------------------------------+-------------+
|                                | VIEW        |
|                                | [x] Top-down|
|                                | [x] Grid    |
|                                | [ ] Tags    |
|         video or snapshot      | SNAPSHOT    |
|         (name menu on frame)   | [Find+Name] |
|                                | [Bases only]|
|                                | [Accept][Un]|
|                                | REFERENCE   |
|                                | [Take][Clr] |
|                                | STATUS/DEBUG|
+--------------------------------+-------------+
```

- Good: Smallest work. All functions are visible. The name menu and its tests stay. Camera view about 1580 x 890.
- Bad: Controls that do not apply stay on the screen (disabled). The panel gets longer with each priority. It has no good place for selection data or wounds.

### Option 2: context inspector (recommended)

A short fixed part on top (status and view toggles). Below it, the panel changes with the state: live, snapshot, or selected base. The states are the states that `_Live` has now: `result is None`, `result` set, `session.selected` set.

```
                               Live            Snapshot        Base selected
+----------------------+-------------+  +-------------+  +-------------+
|                      | 4 corners OK|  | 4 corners OK|  | 4 corners OK|
|                      | [x]Top [x]Gr|  | [x]Top [x]Gr|  | [x]Top [x]Gr|
|   video or snapshot  |-------------|  |-------------|  |-------------|
|                      | [Find+Name] |  | 7/8 named   |  | 32 mm base  |
|                      | [Bases only]|  | list of     |  | [crop image]|
|                      | Reference:  |  | operatives  |  | Boss   0.81 |
|                      |  set [Clear]|  | [Accept all]|  | Burna  0.74 |
|                      |             |  | [Undo][Back]|  | Later: rings|
|                      | > Debug     |  | > Debug     |  | and wounds  |
+----------------------+-------------+  +-------------+  +-------------+
```

- Good: Only the controls that apply are on the screen. The name menu becomes a real list with crop and scores. The list of operatives in the snapshot state covers most of Option 3 at low cost. Priority 5 goes in "Base selected", priority 6 adds the map lists at the top, priority 7 adds "Token selected". Camera view about 1580 x 890.
- Bad: The controls move when the state changes, so a user must learn the states. The name menu moves off the frame, so the eyes go from the base to the panel.

### Option 3: two team rosters

The two teams are two columns, one each side of the board. A top bar holds the actions. A bottom strip holds the status.

```
+----------------------------------------------+
| [Find+Name][Bases] [x]Top [x]Grid  [Ref] ... |
+---------+--------------------------+---------+
| TEAM A  |                          | TEAM B  |
| Boss  v |                          | Sarge v |
| Burna ? |     video or snapshot    | Gun   v |
| Slasha  |                          | Comms ? |
| ...     |                          | ...     |
+---------+--------------------------+---------+
| 4 corners OK | calibrated lens | message     |
+----------------------------------------------+
```

- Good: Best match for a two-player game. Each player reads their side. Wounds and statuses have a natural place. Operatives that the app did not find are visible. A fix is: click a base, click a row. Top-down view is not smaller.
- Bad: Camera view about 1400 x 790, the smallest here. It needs two teams: with the default (all gallery teams) there can be more than two, so it needs a team choice. Today there is no game state to fill the rows beyond the name. Much of its value comes only with wounds and statuses, which have no priority number yet.

### Option 4: full-window video with floating controls

The video fills the window. Small translucent bars float over it. A drawer opens on demand.

```
+----------------------------------------------+
| (o) 4 corners OK                   [=] drawer|
|                                              |
|              video or snapshot               |
|                        +---------+           |
|                        | name    |           |
|                        | menu    |           |
|  [Find+Name] [Bases] [Top] [Grid] [Ref]      |
+----------------------------------------------+
```

- Good: Largest image, about 1920 x 1080 in camera view. Clean look on a monitor at the table. Keeps the eyes on the board.
- Bad: In camera view, the bars cover parts of the board or the corner cards. Top-down view has free space that this option does not use. Most custom widget work and styling. Less space for lists. Debug data is one more click away.

### Option 5: workspaces (Setup, Identify, Play)

A top switch selects one of three workspaces. Each has its own panel. It follows the order of a game.

```
+----------------------------------------------+
| [ Setup ]  [ Identify ]  [ Play ]     status |
+--------------------------------+-------------+
|                                | Setup:      |
|                                |  corners    |
|         video or snapshot      |  reference  |
|                                |  doors      |
|                                | Identify:   |
|                                |  names      |
|                                | Play:       |
|                                |  rings,wound|
+--------------------------------+-------------+
```

- Good: Guides a new user through the game order. Setup (map choice) and game steps fit it. Each panel stays short.
- Bad: Today the Setup workspace has only the reference, and Play is empty until priority 5. More clicks during a game: a snapshot and a fix can be needed each turn. It needs a mode state that the core does not have.

### Comparison

| | 1 Panel | 2 Inspector | 3 Rosters | 4 Floating | 5 Workspaces |
|---|---|---|---|---|---|
| Work for parity with today | Small | Medium | Medium to large | Large | Medium |
| Camera view size | 1580 x 890 | 1580 x 890 | 1400 x 790 | 1920 x 1080 | 1580 x 890 |
| Fits priorities 5 to 7 | Poor | Good | Good for 5, poor for 7 | Medium | Good |
| Fits wounds and statuses | Poor | Medium | Best | Poor | Medium |
| Clicks in one turn | Fewest | Few | Few | Few | More |
| Useful before priority 5 | Yes | Yes | Partly | Yes | Partly |

The work ratings are estimates.

Add-on for options 1, 2, 3 and 5: a small live inset over the snapshot. A snapshot stops the video, so the inset shows that the camera still works and shows a bumped board. It does not replace the drift status. It needs a second image from the worker per frame.

## 6. Toolkit

The plan selects PySide6 and a monitor or laptop next to the table. This file keeps that decision. Options 1 to 5 are possible in a browser too, but with about 1.5 to 2 times the front-end work (JPEG video transport, state sync, three languages). Start with a browser only if a tablet at the table becomes a goal for the MVP.

OpenCV-drawn panels are not an option. They need hand-made hit tests and layout for each control, and text entry and lists are not practical.

## 7. Code structure (same for all options)

All items are implemented. Section 13 has the names in the code.

1. `src/battlesight/live.py`, no Qt. A `LiveSession` class gets the logic of `cmd_live`: `_Live`, `_Notice`, `_BoardWindow`, `snapshot`, `render_live`, `render_snapshot`, gallery lazy load.
   - `step(frame) -> LiveFrame`: image, and the JSON-safe state dict of section 8. A later web front end can use the same dict.
   - The image has only the board overlays (tags, grid, bases, names). The on-frame strips and the name menu go away from the image.
   - One action per function of section 3, in the action list of section 8.
   - Coordinates: a click gives image pixels. The front end converts widget pixels to image pixels. `LiveSession` converts to board mm.
2. `src/battlesight/ui/`, PySide6. The shell, the shared parts and `layouts/` (section 8). Import it only inside `cmd_live`, so that other commands and tests do not load Qt.
3. Threads. One worker `QThread` owns the camera, the `LiveSession` and the `Pipeline`. The GUI thread sends commands to a queue. The worker runs them between frames, so no locks are necessary. The worker sends each `LiveFrame` to the GUI thread with a signal. The panel shows "busy" during a snapshot, and the window stays usable. The video stops during the snapshot, as it does now.
4. Keys. The shell makes one `QShortcut` per action, with the same keys as now. Shared parts must have `Qt.NoFocus`, or `space` pushes the focused button.
5. Speed. The worker resizes the frame to the widget size before the conversion to `QImage`. Drop old frames: send a new frame only after the GUI has painted the last one.
6. Camera down. `Camera.read()` gives `None`. The worker sends a status and continues. The panel shows the camera state.

`cmd_live` becomes a short function: parse, load, open the window. The `imshow` loop goes away. There is no fallback mode and no new option. The other commands keep `cv2.imshow`.

## 8. Modular layouts

The app has many layouts on one game core. A layout only places widgets. It has no game logic and no game state. A setting selects the layout, and the layout can change while the app runs.

### Layers

| Layer | Where | Holds | Uses Qt |
|---|---|---|---|
| Game core | `live.py`, `LiveSession` | State and logic. Runs on the worker thread. | No |
| Contract | `live.py`, action list and state dict | What the user can do and what the user can see. | No |
| Shell | `ui/shell.py` | Window, worker, keys, generated menu, layout change. | Yes |
| Shared parts | `ui/parts.py` | `VideoWidget`, action button, operative list, candidate list, status view, debug view. | Yes |
| Layouts | `ui/layouts/*.py` | Placement of the parts. Layout options. | Yes |

### Contract: actions

Each function is one registered action. A new priority adds its actions to this list, one time.

```python
Action("snapshot.name", label="Identify operatives", key="space", enabled=lambda s: not s["busy"])
Action("view.grid",     label="Grid",          key="g",     checked=lambda s: s["grid"])
Action("base.choose",   label="Set name",      key=None,    enabled=lambda s: s["selected"] is not None)
```

- A layout calls `shell.dispatch("snapshot.name")`. An action with a value takes arguments: `shell.dispatch("base.choose", label=..., index=...)`, `shell.dispatch("board.click", x=..., y=...)`. `MainWindow.dispatch` drops a command while `busy`, and returns the command number or `None`.
- `dispatch` puts the command on the worker queue (section 7).
- `Action.enabled` and `Action.checked` only tell a front end how to show a control. `LiveSession.dispatch` does not read them, so a key on a disabled control still gives its message (for example "push space first").
- `Action.args` names the values of an action. An action with `args` has no menu item. An action with `shell=True` (`app.help`, `app.quit`) is done by the front end, and the core has no handler for it.
- Action ids are strings, so a later web front end can send them as messages.

### Contract: state

The state dict is JSON-safe. The worker sends it with each frame.

| Key | Meaning |
|---|---|
| `mode` | `live` or `snapshot`. |
| `busy` | A snapshot or a model load is running. |
| `top_down`, `grid`, `terrain`, `zones`, `keys` | View settings. |
| `drift` | Status value, summary text, lost tags, moved card. |
| `camera` | Camera up or down. |
| `lens`, `teams`, `reference_set` | Facts for the status view. |
| `snapshot_ok` | All stages of the snapshot passed. `null` in the live view. |
| `naming` | The snapshot has names that the user can fix. |
| `bases` | One entry per base: `index`, `diameter_mm`, `label`, `score` (`null` when not finite), `sure`, `confirmed`, `position_px` (pixels of the returned image). |
| `selected` | Index of the selected base, or `null`. Also without names (after `b`). |
| `target` | Index of the second base of a distance, or `null`. |
| `ring_in` | Ring on the selected base, in inches (1 to 9), or `null`. |
| `distance` | Rim to rim from `selected` to `target`: `{mm, in}`, or `null`. Layouts show only the inches, for example `5.93"`. |
| `candidates` | Names for the selected base: list of `{label, score}`. |
| `can_undo`, `unconfirmed` | Values for the undo and accept controls. |
| `notice` | Last message for 6 s: `{text, level}`, level is `error`, `hint` or `done`. `null` when there is none. |
| `debug` | `tags`, `fit_max_mm`, `fps`, `detect_ms`. Values are `null` when not known. |

`drift` is `null` until the first frame. Its keys are `status` (the `DriftStatus` name), `ok`, `text`, `lost`, `card`. `top_down` is the view on the screen: the snapshot view in a snapshot, else the live view.

Base positions are in the state, so a layout can put a popup or a label at a base. The core does not draw layout items such as popups and hover marks.

Crop images and the video frame are not in the dict. The `LiveFrame` holds them next to the dict.

### Layout interface

```python
class Layout(Protocol):
    name: ClassVar[str]
    def build(self, shell: Shell, options: dict) -> QWidget: ...  # make and place the parts
    def on_state(self, state: dict) -> None: ...                  # show the new state
```

A layout also has `name_popup: ClassVar[bool]`: `True` means that the shell opens its name popup at a right-clicked base. `shell` gives a layout `dispatch(id, **args)`, `new_video()` and `selected_crop()` (the crop of the `LiveFrame` that `on_state` shows). `build` calls `new_video()` one time: the shell connects the widget and keeps it as `self.video`. `build` raises `ValueError` for a bad value in `options`. The shell then shows a notice and builds the layout again with no options.

`ui/layouts/__init__.py` holds a plain dict from name to class. There is no plugin discovery.

The shell gives all layouts these items:

- A key for each action.
- A menu bar made from the action list. A new action is available in all layouts with no layout work.
- A default name choice: a `QMenu` popup at a right-clicked base (`popup`, not `exec`), from `selected` and `candidates`. A layout can replace it with its own list.

Selection stays in the core, so it stays after a layout change. A layout can keep its own small UI state, for example an open drawer. That state goes away at a layout change.

### Layout setting

The option rule applies: the layout can change while the app runs, so it has no command option.

- Change it with the `View > Layout` menu or the `l` key (`app.layout`, next layout).
- The change replaces the central widget. Each layout makes its own parts, and the shell removes the old ones. The worker, the camera, the snapshot and the names continue.
- `data/ui.toml` keeps the choice. It is local to the machine, and git ignores it. The shell writes only `[ui] layout` and keeps the other values, as the writers of `data/camera.toml` do.

```toml
[ui]
layout = "inspector"

[layouts.inspector]
panel_side = "right"
debug_open = false
```

- A missing file is not an error: the app uses the default layout.
- An unknown layout name or a bad layout value must not stop the app. The app uses the default and shows a notice.

### Other front ends

A browser front end later uses the same action ids and state dict over a WebSocket. It starts in a different way, so a change between Qt and browser needs a restart. A change between Qt layouts does not.

### Rules

- A layout does not import `pipeline`, `ident`, `calib` or `camera`.
- A layout does not change the state dict.
- A layout shows the actions that it selects. All others stay available through the keys and the generated menu.

### Build order

1. `panel` (Option 1). The shell generates its controls from the action list. It is cheap, it always has all functions, and it proves the contract.
2. `inspector` (Option 2), the main layout. It proves that the contract carries selection and state-driven panels.
3. `rosters` (Option 3), when wounds and statuses exist.
4. Options 4 and 5 only if the first layouts show a need. Each layout adds maintenance at each priority.

## 9. Risks

- Exit on macOS. Open issue, section 13.
- Contract growth. Each new field and action must serve all layouts. Keep the state flat and small. A contract test (section 11) stops a layout that does not follow it.
- Qt and OpenCV. Checked on this Mac: the `opencv-python` wheel has the COCOA GUI backend and no Qt plugins, so there is no plugin conflict. On Linux the wheel can have its own Qt plugins. Only one GUI toolkit runs in each command, so `cmd_live` must never call `cv2.imshow` or `cv2.waitKey`.
- Size. The app uses `PySide6-Essentials`. It has all the modules that the UI needs today.
- Detection time. The worker runs tag detection on each frame, and commands wait for the end of the frame. Measured on a 12 MP photo: about 180 ms per frame. A key took more than 0.5 s to show one time. A real 4K frame has fewer pixels, but it is not measured yet. If the delay is too long, run detection on a smaller copy of the frame.

## 10. Decisions

1. Layouts: `panel` and `inspector` are built. `inspector` is the default.
2. The name fix in `inspector` is a list in the panel, with crop and scores. `panel` uses the shell popup.
3. Live inset over the snapshot: not in the first version.
4. PySide6 now. A browser front end can come later, if a tablet at the table becomes a goal.

## 11. Done test for the UI

- `battlesight` with no subcommand opens the window. It has no new option.
- Each row of section 3 works with its key and with a control, in each layout. The control can be in the generated menu.
- The menu and the `l` key change the layout while the app runs. The camera does not restart. A snapshot, its names and the selected base stay.
- The app opens with the layout in `data/ui.toml`. A missing file, an unknown layout or a bad value does not stop the app.
- Contract test, offscreen, for each layout: `build` works, and `on_state` works for each state of a recorded list (live, busy, snapshot, base selected, camera down, each drift value).
- A test fails when a layout module imports `pipeline`, `ident`, `calib` or `camera`.
- The video has no status text. The status area shows all five `DriftStatus` values and "board position updated".
- The GUI does not stop during the first `space` (model load).
- A closed camera shows a status and the camera comes back with no restart.
- A click on a base in a 4K frame selects the right base in both views, also after a window resize.

## 12. Steps after the decision

These steps are the UI foundation. It comes before priority 5. `specs/plan.md`, section 2, "UI work per priority" has the UI part of each priority after that.

1. Done. Move the logic to `LiveSession`, with the action list and the state dict. Keep the `imshow` loop on top of it, and run all tests. This step does not change behaviour.
2. Done. Add the shell, the worker, the keys and the generated menu. Replace the `imshow` loop. Status text is in the status bar. Commands are dropped while a snapshot runs.
3. Done. Add the `panel` layout, `data/ui.toml`, and the contract test.
4. Done. Add the debug values.
5. Done. Add the `inspector` layout and the layout change (`View > Layout`, `l`).
6. Done. Update `README.md` ("How to use", "What you see"), `docs/setup.md` if a user sets `data/ui.toml` by hand, `docs/cli.md` (live app, keys, `data/ui.toml`), `docs/testing.md` (contract test), `CLAUDE.md` (architecture), and `.gitignore`.

## 13. Notes for the next implementer

Status on 2026-11-08: all steps of section 12 are in the work tree. They are not committed.

### Code that exists

| Name | Where | Use |
|---|---|---|
| `LiveSession.step(frame)` | `live.py` | One frame. `None` = camera down. Returns `LiveFrame(image, state, scale)`. |
| `LiveSession.dispatch(id, **args)` | `live.py` | One action. An unknown or a `shell` action is a `KeyError`. |
| `LiveSession.current()` | `live.py` | Last image with the state of now. |
| `LiveSession.set_display_width(w)` | `live.py` | Largest width of the returned image. |
| `LiveSession.on_busy` | `live.py` | Callback before a snapshot. The worker sets it, so that the GUI gets `busy` before the long work. |
| `Worker` | `ui/shell.py` | `QThread`. `submit(id, **args)` returns the command number. `frame_ready(live, done)` gives the number of commands done. |
| `MainWindow` | `ui/shell.py` | `dispatch(id, **args)`, `trigger(action)`, `new_video()`, `set_layout(name)` (changes the layout and saves the choice), `current_layout`, `shortcuts` and `menu_actions` (by action id), `video`, `state`, `last_frame`, `popup`. |
| `Worker.notify(text)` | `ui/shell.py` | Error notice from the GUI thread. The worker gives it to the session. |
| `LAYOUTS`, `DEFAULT_LAYOUT`, `Layout`, `Shell` | `ui/layouts/__init__.py` | The layout list and the interface. |
| `PanelLayout` | `ui/layouts/panel.py` | Controls from `ACTIONS`. Option `panel_side` (`left`, `right`). |
| `InspectorLayout`, `page_for`, `summary_text` | `ui/layouts/inspector.py` | Pages `live`, `snapshot`, `selected`. Options `panel_side`, `debug_open`. |
| `MapPicker`, `DebugView`, `OperativeList`, `CandidateList` | `ui/parts.py` | Shared parts. A list fills again only when its data changes. |
| `LiveFrame.crop` | `live.py` | The board around the selected base, or `None`. |
| `action_control`, `action_group`, `show_action_state` | `ui/parts.py` | One control for one action. |
| `load_ui_settings`, `save_layout`, `DEFAULT_UI_FILE` | `ui/settings.py` | `data/ui.toml`. No Qt. `save_layout` edits only the `[ui] layout` line. |
| `status_text(state)`, `help_text()` | `ui/shell.py` | Pure functions. |
| `run_window(session, camera)` | `ui/shell.py` | Opens the window. Tests replace it with `monkeypatch`. |
| `VideoWidget` | `ui/parts.py` | `set_image`, `clicked(x, y, global_pos)`, `width_changed(px)`. |
| `ImageSource` | `camera.py` | One image file as a camera (`--image-source`). |

Actions that section 3 does not list: `view.terrain`, `view.zones`, `view.keys`, `map.terrain_type(name)`, `map.layout(name)` (no keys), `base.deselect` (no key), `board.click(x, y)` (left click), `board.select(x, y)` (right click: replaces the selection), `base.choose(label, index=None)`, `ring.1` to `ring.9`, `ring.clear`.

The live view (`LiveSession.top_down`) and the snapshot (`BoardWindow.top_down`) have separate views. A snapshot always opens top-down. `v` keeps the selected base, the ring and the distance, because they are in board mm. The old `_Live`, `_Notice`, `_BoardWindow`, `overlay.Menu` and `HELP_KEYS` no longer exist.

### Rules that the layouts need

- `set_layout` closes the shell popup, and the selected base stays. After a change to `panel`, the base stays selected with no popup. `Esc` clears it, and a right click opens the popup.
- An action that a layout selects from a list sends `index` with `base.choose`.
- A row of `OperativeList` sends `board.click` at `position_px`. There is no separate selection action.

### Traps that were found

- **Keep the frame flow alive.** The worker sends a frame only after `gui_free()`, which is the last line of `_on_frame`. An exception before that line stops the video and leaves no error on the screen. `on_state` of a layout runs there, so `gui_free()` is in a `finally` block. Keep it there.
- **Popup order.** A popup sends `index` with `base.choose`, so that the result does not depend on the selection when the command runs. Keep this for each list that selects a name.
- **Click and popup.** A `LiveFrame` can be older than a click. `_on_right_click` keeps the command number, and the popup opens only when `done` has reached it. Use the same check for each result of a click.
- **Stale click positions.** A click uses the image on the screen. If the view changed in the last frame, the core can miss the base. It happened with a scripted click about 0.5 s after `v` on a 12 MP photo. It is not fixed. A user can click again.
- **Width.** `fit_to_width` only makes an image smaller, and it knows only the width. `VideoWidget` scales the rest, for example a portrait photo in a wide window.
- **Saved reference.** With `--image-source`, the app does not load `empty_board.png`, because it is from another camera view. Without `--reference`, `e` does not write a file (`reference_path = None`).
- **Gallery.** A name fix on a debug image adds a crop to the real gallery. Use another `--gallery-root` for tests.

### Tests

- `tests/test_ui.py` runs offscreen and has `window`, `press`, `mouse_click`, `widget_pos`, `wait_until`. Set `QT_QPA_PLATFORM` before the first PySide6 import. Wait for a state with `wait_until`. Do not use fixed delays.
- `test_bases_clear_rims_found[tomb-world-2.jpg]` fails before and after the UI work. It is not a UI failure.
- Offscreen tests do not show Cocoa behaviour. Also try a real window on a photo, as below.

### How to try a real window with a script

- Monkeypatch `MainWindow.start` to schedule the input with `QTimer.singleShot`.
- Press keys with `window.shortcuts[id].activated.emit()`. `QTest.keyClick` works only when the window has focus, and macOS does not let a script take the focus from another app. Log `isActiveWindow()`.
- For a gallery without the model, use `dataclasses.replace(cfg, ident_model="color-hist")`.
- Call `faulthandler.dump_traceback_later(30, exit=True)`, and write the output to a file, not to a pipe. A hung process then ends and shows its stack.
- A first `space` on a 12 MP photo took more than 3 s with the colour-histogram model. Wait for the state.

### Open issue: exit on macOS

After the window closed, `app.exec` sometimes did not return, and the Terminal stayed stuck. The worker had stopped, and the only Python stack was in `app.exec`.

| Exit code in `run_window` | Result in real windows |
|---|---|
| Automatic quit after the last window | 2 of 5 runs hung, with focus. |
| `app.quit`, queued | 4 of 8 runs hung, in the runs where the script could use the window and the name popup opened. 0 of 20 hung without focus. |
| `app.exit` through `QTimer.singleShot`, and `closeEvent` closes the popup | 0 of 8 hung, without focus. |

The `app.exit` fix is not proven, because a hang was seen only with focus. The cause is not proven either. The likely cause is that `app.quit` goes through `NSApp terminate` on macOS. To test: focus the window, use the name popup, push `q`, and check that the Terminal comes back. If it hangs again, run `sample` on the Python process, not on its `uv` parent.

### Not yet tried

- A real gallery and the DINOv2 load in `inspector`. On 2026-11-08 the GUI gap was 0.34 s during the first `space` with a colour-histogram gallery. `data/teams` was empty, so the model load was not measured.
- A real camera with the Qt window. The camera opens on the main thread and reads on the worker thread. `Camera` also reconnects on the worker thread. Check Camo (AVFoundation) and a network stream.
- The real DINOv2 model load in the window (the done test "the GUI does not stop during the first `space`"). The tests use the colour-histogram model.
- The latency on a real 4K frame.
