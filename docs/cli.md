# Battle Sight CLI

Run all commands with `uv run` from the repository root.

```
uv run battlesight --help
uv run battlesight <command> --help
uv run battlesight --version
```

## All commands

| Command | Input | Use |
|---|---|---|
| `battlesight` (no command) | live camera, or one image for debugging | The app. Tracks the corner cards, shows the camera or the board from above with the grid, finds the bases and names the operatives on a key push. |
| `battlesight enroll` | live camera or saved photos | Add gallery photos of one operative. |
| `battlesight gallery` | gallery folder | List the gallery and check each photo against the rest. |
| `battlesight calibrate` | live camera or saved photos | Calibrate the lens with the ChArUco sheet. Write the `[lens]` table of `data/camera.toml`. |
| `battlesight image-pipeline` | saved photos | Run all stages on photos. Write a JSON report and overlay images. Use it for tests and automation. |
| `battlesight map` | map files | Check terrain map files and draw each on an empty board. |
| `tools/list_cameras.py` | cameras | Find the camera and save its source, frame size and frame rate to `data/camera.toml`. |
| `tools/make_tags.py` | none | Print sheet of the corner cards: `data/print/corner_tags.pdf`. |
| `tools/make_charuco.py` | none | Lens calibration sheet: `data/print/charuco.pdf`. |
| `tools/bench_bases.py` | test photos | Bases benchmark: metrics, parameter sweeps, baseline check. |
| `tools/bench_ident.py` | test photos | Identification benchmark: name metrics, baseline check. |
| `tools/annotate_bases.py` | one test photo | Click the base ground truth of a test photo, and assign the operative names. |

## Option rule

The commands have as few options as possible. A feature that a key (later a
UI button) turns on or off, or a mode or view that a key changes, has no
option. Options are only for values that do not change while the app runs.

## Camera values

The camera values are not command options. They are in `data/camera.toml`
(see "Camera file" below):

| Value | Where to change it |
|---|---|
| Source (device index or stream URL) | `tools/list_cameras.py --use INDEX` or `--url URL`. For one run: `--source`. |
| Frame size | `tools/list_cameras.py --use INDEX --width W --height H`. A stream gets its size from the phone app. |
| Frame rate | `tools/list_cameras.py --use INDEX --fps N`. |
| Lens | `battlesight calibrate` (`[lens]`), or `focal_35mm` by hand for the nominal lens. |

You can also edit the `[camera]` table in the file with a text editor. The
windows are resizable: drag the window edge to change the size on screen.
The analysis always uses the full camera frame.

## `battlesight`: the live app

```
uv run battlesight [--team NAME [NAME2]] [--gallery-root data/teams]
                   [--reference data/snapshots/empty_board.png]
                   [--source S | --image-source FILE]
                   [--camera-file data/camera.toml] [--config data/board.toml]
```

| Option | Default | Meaning |
|---|---|---|
| `--team` | all teams in `--gallery-root` | One or two team folders (two teams play). Labels are `<team>/<operative>`. More than two names: exit 2. Fills the two team lists of the window at start; the user can change them while the app runs (`team.set`). |
| `--gallery-root` | `data/teams` | Folder with one folder per team. |
| `--reference` | `data/snapshots/empty_board.png`. With `--image-source`: none. | Empty-board reference. Loaded at start when the file exists. The `e` key writes it. Git ignores the file (`data/snapshots/*.png`). |
| `--source` | from the camera file, else `0` | Camera index (Camo, built-in camera) or network stream URL (`http://...`, `rtsp://...`), for this run only. A stream is read on a background thread that keeps only the newest frame. See [setup.md](setup.md), step 4. |
| `--image-source` | none | One image file to use instead of the camera, for debugging and tests. See "Image source" below. Not with `--source`, and not with a command. An image that the app cannot read: exit 2. |
| `--camera-file` | `data/camera.toml` | Saved camera settings. A missing file is not an error: the built-in defaults apply (camera 0, 1920 x 1080, nominal 24 mm lens). |
| `--config` | `data/board.toml` | Board and tag config. |

### Image source

```
uv run battlesight --image-source tests/images/tomb-world-1.jpg
```

The app reads one image file and uses it as the camera. It gives the same frame
30 times per second (`camera.ImageSource`), so the drift status locks and all
keys, clicks and views work as with a camera. The app applies the EXIF
orientation of phone photos, and it accepts the image types of
`image-pipeline`. To use a frame of the real camera, push `s` in the live app
and give the saved file from `data/snapshots/`.

To get a report and no window, use `image-pipeline`.

| Item | With `--image-source` |
|---|---|
| Lens | From `--camera-file`, as with a camera. The calibration applies only when the image has the same aspect ratio. Else the nominal lens applies. Use `--camera-file` with a missing file to get the nominal lens, as in the photo tests. |
| Reference | The saved `empty_board.png` comes from the live camera, so the app does not load it. To use a reference, give `--reference FILE`. Without `--reference`, the `e` key sets the reference for this run only and writes no file. With `--reference`, the `e` key writes that file. |
| Gallery | A name fix, `a` and `u` change the real gallery in `--gallery-root`. To keep the gallery clean during a test, give another folder. |
| `s` key | Writes a copy of the image to `data/snapshots/`. |

### Live view

On every frame the app finds the four corner tags and runs the drift
monitor. By default it shows the board from above at `rectify.px_per_mm`, with
the lens distortion removed and the map grid on top (see "Map grid" below).
The UI calls this "Adjust perspective". `v` changes to the camera frame (tags,
board corners, grid). The camera frame also shows when the app has no board
fit, for example when a tag is not visible. The status bar under
the image shows the drift status and the lens in use: `calibrated` (from
`[lens]`) or `nominal`, and the result count of a snapshot. It has no key
hints. The video has no status text.

Drift status (`calib.DriftMonitor`):

| Status | Meaning |
|---|---|
| `locking reference` | The app saves the tag positions after `drift.settle_frames` still frames with all four tags. |
| `4 corners OK` | All tags are within `drift.tolerance_mm` of the saved positions. The top-down view uses the saved positions, so it does not jitter. |
| `BOARD MOVED` | One or more tags moved more than `drift.tolerance_mm`. The view follows the tags in each frame. |
| `board position updated` | After a move the tags were still for `settle_frames` frames, and all four cards moved as one piece (the board or the camera moved). The app saved the new positions by itself. |
| `CARD MOVED` | After a move, one card moved relative to the other three by more than `drift.tolerance_mm`: it is no longer on its board corner. The status stays until the card is back. |
| `TAG LOST` | A tag was not seen for `drift.lost_after_frames` frames. With one or two tags out of view (`calib.HOLD_MAX_LOST`) and the other tags within `drift.tolerance_mm` of the saved positions, the report is `held`: the view keeps the saved positions. With three or more tags out of view, or a moved tag in view, there is no board fit. |

The one-card check (`calib.card_offset_mm`): for each tag, a homography from
the other three tags (12 corners, saved -> now) predicts its corners. When
the board or the camera moves, all four predictions fit. A bumped card is
off by its shift.

### Snapshot

`space` runs all stages (`corners`, `rectify`, `bases`, `ident`) on one
frame. `b` runs only up to `bases`: no names, and the gallery and the model
do not load. The snapshot stays on the screen until `Esc` ("Go back to
Live"). Then the live video shows again, and the bases, names and selection
of the snapshot stay on it, in board mm, until the next `space` or `b`
(`LiveSession.frozen` is the mode, `result` the last snapshot). Clicks, rings
and distances work on the live video too, while it has a board fit. Without
a board fit (tags lost, camera view) `position_px` of each base is `null`.
The command prints each base to stdout.

The gallery and the model load on the first `space`, not at start. No team
folder: the app says so, and the bases have no names.

The snapshot shows each base with its operative name on a dark box (no score
in the live app; `image-pipeline` overlays keep the score). In the live app,
with teams chosen (`team_slots`), the circle and the name have the team
colour (`live.TEAM_COLORS`: left red, right blue; a base with no name grey,
`overlay.draw_idents(team_colors=...)`). Else: green =
sure, yellow = unsure (the lead over the second name is less than
`ident.min_margin`, or the one-to-one assignment gave the base its second
choice), red `?` = unknown (score under `ident.min_score`, or no operative
of this base size is left).

A snapshot opens in the top-down view. `v` changes its view. The live view
keeps its own view. The selection, the ring and the distance stay when the view
changes.

While a snapshot runs, the status bar shows `working ...`, the window stays
usable, and the app ignores keys and clicks. Only `q` works: the window closes
when the snapshot ends.

Over a snapshot, the status bar still shows the live drift status (for example
`TAG LOST`, `CARD MOVED`). Messages of keys and clicks (a failed stage, no
base found, nothing to undo, a name set, a photo added) print to the Terminal
and also show for 6 seconds at the right of the status bar: red for an error,
yellow for a hint, green for done. `live.Notice` holds the last message.

Rings and distances (plan priority 5, `measure.py`) work after `space` and
after `b`. A left click on a base selects it. A left click on another base
makes it the target: the app draws the shortest line between the two rims, and
the status bar shows the distance in inches (`5.93"`). Another base moves the target, and a click
on the target removes it. A click on the selected base or on the empty board
clears the selection. Keys `1` to `9` draw a ring with radius = base radius +
N in on the selected base, and `0` removes it. A new selection starts with no
ring and no target. The ring keeps to the selected base and is sampled in
board mm, so it also bends with the lens in the camera view. Touching or
overlapping bases have a distance of 0.

Measurement around walls (`walls.py`, action `measure.walls`, key `w`, check
box "Measure around walls"; enabled only with a layout, on at each layout
load). The obstacles are axis-aligned rectangles from the terrain shapes
(`walls.obstacles`): walls, pillars, wall ends, closed hatchways and breach
points. An open opening cuts a gap of its width in its wall. The measure line
has no width: it can touch a corner or run along a wall face. Other bases do
not block it. Terrain on an outer grid line (`BoardConfig.grid_lines_mm`)
blocks as far as the board edge (its rectangle goes 1 mm past the edge), and
a path does not bend at a corner that is off the board: the printed margin
between a pillar and the board edge is no way around it.

- Distance: `Navigator.path` gives the shortest path between the two rims. A
  clear centre line gives the same result as `measure.rim_gap`. Else Dijkstra
  runs on the visibility graph of the rectangle corners (exact lengths; a base
  links to a corner through 64 rim points and the rim point on the line to
  its centre). `state["distance"]` is `{"mm", "in", "around"}`; with
  `around = true` and no path, `mm` and `in` are null and the UI shows
  `no path`.
- Ring: `Navigator.reach` gives the outline of all points with a path length
  of N in or less. Sources are the base centre, 16 rim points and each corner
  nearer than the range; each source fills a disc with the range that is
  left, without the shadow of each rectangle. The union is a mask at
  `walls.RANGE_PX_PER_MM` (4), its contours go back to board mm (a contour in
  another one is a hole). `overlay.draw_range_mm` fills it and draws the edge.
Movement (`moves.py`, action `measure.move`, key `m`, check box "Measure
movement" next to the ring control; works with and without a layout). With it
on, the ring of N in is the area that the selected base can move to, and the
front end sends the pointer position (`board.hover`, `x` / `y` in image pixels
or null) for the route to the pointer. The distance to a target does not
change.

- Rule: a move is a chain of straight steps of 1 in or less, each step counts
  as 1 in. Thus a straight increment counts as its length rounded up.
- `moves.move_field` makes a raster (`MOVE_PX_PER_MM` = 1) of the positions
  of the base centre. `stand`: board inset by the base radius, without the
  terrain rectangles grown by the base radius (`SLACK_MM` = 0.5 less) and
  without the enemy discs (radius = both radii), and without each open gap rectangle grown by the base radius (no stop on an open hatchway or breach point). `free` adds a zone around
  each open gap (`walls.open_gaps`, base radius + 3 mm around the gap line,
  without the terrain itself): the base can be over the wall parts there
  while it moves, but it cannot stop. A line in the gap (`barrier`) costs
  1 in more to go over.
- Field: level N + 1 = all points that a point of level N sees within 1 in
  (rays on the raster, `_visible`); far from terrain a plain dilation. Only
  the edge of level N is a source. A step over a gap line gives level N + 2.
  One straight line from the start gets the exact length rounded up
  (`_seen_from`). A base that starts a little in terrain gets out over
  `ESCAPE_MM`.
- `MoveField.outline`: reached `stand` positions, grown by the base radius
  (the area that the base covers), contours in board mm.
- `MoveField.route(point)`: goes back level by level to the start, then picks
  the chain positions that give the fewest inches and the fewest increments.
  An increment up to `ROUND_SLACK_IN` (0.06) over a full inch counts as that
  inch. A point on the covered area that the centre cannot get to goes to the
  nearest position in reach.
- Enemies: bases with a name of another team than the selected base
  (`LiveSession._enemies`). No name on the selected base: nothing blocks.
- All other bases are `friends` of `move_field`: they are taken out of
  `stand` only (disc of both radii), so a move goes through them but cannot
  stop on them. The outline has a hole at each one, and the pointer on one
  gives no route.
- `state["movement"]`, `state["move"]` = `{"in", "increments"}` or null. The
  status bar shows `move 5" (3 + 2)`. Layout 6 at 9 in: about 0.15 s for a new
  field, about 2 ms for a route. The shell sends one pointer position at a
  time and keeps only the newest (`MainWindow._on_hover`).

Control marks: `view.marks` ("Control marks", on at the start, no key,
enabled only when a base has a name). After names (`LiveSession.session`), `LiveSession._control`
gives `state["control"]` = `{"melee": [[i, j]], "objectives": [[i, k]]}`
(null with the marks off or no names). `melee`: `measure.melee_pairs`,
named bases of two different teams with rim gap at most 1 in; one
`overlay.draw_melee_sign` (yellow crossed swords on a dark octagon,
`MELEE_SIGN_MM` radius, upright) at the middle of the rim gap of each pair,
drawn after names and health bars. `objectives`: `measure.objective_contacts`,
named bases with rim gap to the objective marker edge (`[terrain]
objective_mm`) at most `objective_range_mm`; index `k` in the objective
shapes of the layout. `overlay.draw_objective_arrows` (drawn under the names)
puts heads every `ARROW_STEP_MM` from the base rim toward the objective
centre, last tip not past the centre; with less than two steps from the rim
to the centre, four heads at 45 degrees outside the rim point to it.
`state["marks"]` is the check box.

- `LiveSession` keeps the navigator per layout and set of open openings, and
  the last path and area per base position, because the live view draws again
  on each frame. Layout 6 at 9 in: about 0.1 s for a new area.

Openings: a left or right click with no base under it selects the hatchway or
breach point of the map layout within 8 mm (`OPENING_HIT_MM`), live and
snapshot. One thing is selected at a time: a base or an opening. State
`selected_opening` = `{index, kind, open}`. `base.deselect` clears both.

False bases: `base.remove` ("Remove ID", no key: `Backspace` and `Delete` edit
the wounds box) removes the selected base from the snapshot (terrain or board
print found as a base). It works with and without names. The later bases move
one index down (`result.scratch` lists and `LabelSession.remove`), and the
selection clears. Gallery photos that a name fix of this snapshot added for
that base are removed too, and `names.undo` skips that fix. The next snapshot
finds the base again.

Game state (`LiveSession`, no window): wounds and openings.

- Wounds: `LiveSession.wounds` = `{label: current}` for operatives that are
  not at their maximum; the maximum is `wounds` of the operative in
  `team.toml` (`ident.Operative.wounds`). The values are by label, so a new
  snapshot and a name fix keep them. `wounds.minus` (`-`), `wounds.plus`
  (`+`) and `wounds.set(value)` change the selected named base, limited to
  0 .. maximum. Without a named base, a hint; without a maximum, an error
  notice (no change). State: `selected_wounds` = `None` (no named base) or
  `{label, current, max}` (`current` and `max` `None` without a maximum);
  each `bases` entry and `rosters` row has `wounds` and `max_wounds`.
  `overlay.draw_health_bars` draws a bar above each named base with a
  maximum: length `HEALTH_MM_PER_WOUND` (2 mm) per maximum wound, fill =
  current / maximum, colour by state: `HEALTH_UNHARMED` (green) at the
  maximum, `HEALTH_WOUNDED` (orange) for maximum / 2 <= current < maximum,
  `HEALTH_INJURED` (red) for current < maximum / 2; red edge at 0.
- Openings: `LiveSession.opened` = indexes in `openings` that are open. A
  layout load (`map.layout`, also the same layout again) closes all.
  `opening.toggle` (`o`) opens or closes the selected opening. State:
  `openings` = `[{index, kind, open}]`. `draw_terrain` draws an open opening
  (`terrain.Shape.open`) as a gap in its wall (the image under the terrain
  shows through) with a jamb in the opening colour across each end. An
  outline alone is not enough: a breach point is as thin as the wall.

Teams: `team.set(slot, name)` sets the team of side 0 or 1 (`None` = none).
The same team on both sides is refused (the team list of the other side
shows it disabled). A change resets the gallery (it loads
again at the next `space`) and the known base sizes; the names of the last
snapshot stay until then. Without a team on either side, `space` uses all
gallery teams. State: `team_choices` (gallery teams), `team_slots`, and
`rosters` (per side `null` or `{team, operatives: [{label, name, base_mm,
base_index, score, confirmed, wounds, max_wounds}]}`, from `team.toml` and the operative folders,
with the base that has the name in the last snapshot). `base.select(index)`
selects a base by index (team list rows).

Fixes (plan priority 4, "user confirms or fixes with one click"): after
`space`, right-click a base, then click a name in the list (a popup at the base
in `panel`, the panel list in `inspector`). The right click replaces the
selection. A name choice clears the selection. The list has the
operatives of this base size, best score first, then the operatives without
gallery photos. The crop of the base goes into the gallery of that operative.
A name that another base held moves to the clicked base; the other base
becomes unknown. A click outside the popup closes it, and the base stays
selected. `Esc` clears the selected base.

### Keys

| Key or mouse | Action |
|---|---|
| `space` | Snapshot: find the bases and name them. |
| `b` | Snapshot: find the bases only. |
| `v` | "Adjust perspective": change between the top-down view (default) and the camera view (live and snapshot). A snapshot opens top-down. |
| `g` | Grid on or off. |
| `e` | Use the current frame as the empty-board reference and write it to `--reference`. The frame must pass `corners` and `rectify`, else the app keeps the old reference and says why. |
| `c` | Clear the empty-board reference: single-frame detection. |
| left click on a base | Select it. With a selected base: set or move the target (second base of a distance). |
| left click on a hatchway or breach point | Select it (no base under the click). |
| `1` to `9` | Ring on the selected base: base radius + N in. |
| `0` | Remove the ring. |
| `w` | Measure around walls on or off (`measure.walls`, only with a layout). |
| `m` | Measure movement on or off (`measure.move`). |
| mouse pointer on the board | With `measure.move` and a ring: the route to the pointer (`board.hover`). |
| `-`, `+` | One wound less or more on the selected named base (`wounds.minus`, `wounds.plus`). |
| `o` | Open or close the selected hatchway or breach point (`opening.toggle`). |
| right click on a base | Select it and show its names (after `space`). |
| left click on a name | Set the name, add the crop to the gallery. |
| `a` | "Accept all identification": accept all names that are not confirmed, add their crops to the gallery. |
| `u` | Undo the last gallery photo of this snapshot (deletes the file, gives the old name back). |
| `s` | "Save screenshot": save the current camera frame to `data/snapshots/snap_<date>_<time>.png`. |
| `l` | Next layout. The choice goes to `data/ui.toml`. |
| `h` | Show the key list. |
| `Esc` | "Go back to Live": show the live video again; the bases, names and selection stay. In the live video: clear the selection. |
| `q` | Quit. |

The README section "What you see" tells what the overlay shows. The game
logic is in `live.LiveSession`: it has no window and no camera. `live.ACTIONS`
lists what the user can do, and `LiveSession.state()` is a JSON-safe dict of
what the user can see (`specs/ui.md`, section 8). The window is PySide6:
`ui/shell.py` has the worker thread (camera and `LiveSession`), the keys, the
menu generated from `ACTIONS`, the status bar, the name popup and the layout
change. `ui/parts.py` has the shared widgets. The other commands keep their
OpenCV windows.

### Layouts and `data/ui.toml`

`ui/layouts/` holds the layouts, in the dict `LAYOUTS`: `rosters` (default),
`inspector`, `panel`. A layout has `name`, `name_popup` (the shell opens its
name popup at a clicked base), `build(shell, options)` and `on_state(state)`. A
layout uses only `shell.dispatch`, `shell.new_video` (one time in `build`),
`shell.selected_crop` and `shell.show_debug(on)` (debug values at the right end
of the status bar; off at each layout change). It must not import `pipeline`,
`ident`, `calib` or `camera`.

`rosters` (Option 3 of `specs/ui.md`): a top bar with the sections View
(`MapPicker`, view toggles and `measure.walls`), Commands (`snapshot.name`, `names.accept` shown
only with `unconfirmed > 0`, a disabled "Add operative", `snapshot.back` shown
only in a snapshot), Inspector (operative: title `Team - Name (40mm)`, crop,
`WoundsControl`, `RingSlider`, `measure.move`, distance, "Change ID" that shows a
`CandidateList`, "Remove ID" = `base.remove`; opening: "Hatchway" / "Breach point" and `OpeningControl`
(state and Open / Close button); `✕` = `base.deselect`), Reference and Debug.
A `TeamRoster` each side of the video; the mark of a row starts with
`current/max` wounds when the operative has a maximum. `WoundsControl`
(`parts.py`, also in `inspector`): `-` and `+` buttons and a `QSpinBox` with
`ClickFocus` (it needs the focus to type; Return or a focus loss sends
`wounds.set` and gives the focus back, so the keys work again); a warning
instead of the controls without a maximum. `inspector` has an `opening` page
too.
A frame from before a check box command (`shell.CHECK_ACTIONS`, `MainWindow._check_sent`)
updates only the image and the status line, not the controls: the old state
must not put the box back for a moment.
`VideoWidget.set_badge(text, color, sign)` paints the mode at the top right of
the image: "Live" with a red `dot`, "Snapshot" with a blue `pause` sign,
"Minimap" with a green `square` (`rosters.shown_mode`: `state["minimap"]`
goes before `state["mode"]`). Scores in all lists are percentages (`parts.percent`).

`data/ui.toml` is local to the machine, and git ignores it. There is no
command option for it. A missing file is not an error. An unknown layout, a bad
file or a bad value does not stop the app: a notice shows, and the default
applies. `ui/settings.py` reads the file. The shell writes only the `[ui] layout` line, and only
at a layout change. It keeps all other text.

| Key | Layout | Values | Default |
|---|---|---|---|
| `[ui] layout` | | `rosters`, `inspector`, `panel` | `rosters` |
| `[layouts.<name>] panel_side` | `inspector`, `panel` | `left`, `right` | `right` |
| `[layouts.<name>] debug_open` | `inspector`, `rosters` | `true`, `false` | `false` |
| `[map] terrain` | | terrain type of the last run (`""` = none) | first type |
| `[map] layout` | | layout name of the last run (`""` = none) | none |

The shell writes the `[map]` lines when the user chooses a terrain type or a
layout in the `MapPicker` drop-down lists, and sends them again at the next
start.

The debug values come from `state["debug"]`: `tags`, `fit_max_mm`, `fps`
(smoothed, of `LiveSession.step`) and `detect_ms` (tag detection of the last
frame).

## `enroll`: gallery photos of one operative

```
uv run battlesight enroll --team NAME --operative NAME [--base-mm 32] [--wounds W]
                          [--images PATH ...] [--reference FILE]
                          [--gallery-root data/teams]
                          [--source S] [--camera-file FILE] [--config FILE]
```

Adds crops of one operative to `data/teams/<team>/<operative>/` and its base
size to `data/teams/<team>/team.toml` (created when new). `--wounds W` (a
whole number, 1 or more) sets or replaces the maximum wounds of the operative
in `team.toml` at the start, before the first crop; a bad value gives exit
code 2 and no change. Live: push
`space`. With exactly one base (of `--base-mm` or the size in `team.toml`)
the crop goes in at once; with more, click the base, or push `Esc` to leave the
choice without a crop (`Esc` outside the choice quits). Turn and move the model
between grabs: 10 to 20 crops from different places on the board (the
parallax changes the look near the edges). A crop with the same pixels as a
gallery photo of the operative is not saved (`ident.DuplicatePhoto`,
`ident.same_photo`): the banner tells the user to turn and move the model. Two
camera frames are never the same, so this stops only a second grab of one
image (`--images`, or a name fix on `--image-source`). `u` undoes the last crop, `v`
changes the view, `q` quits. `e` uses the current frame as the empty-board
reference and writes it to `--reference` (default
`data/snapshots/empty_board.png`, the file that the live app also uses). The
frame must pass `corners` and `rectify`, else the app keeps the old reference
and says why. With a reference, each crop also gets a background crop: the same
square of the empty board at `<operative>/bg/<crop name>`
(`ident.background_path`). It does not depend on the reference file after the
grab. Without a reference there is no background crop and the banner says so.
`t` runs the leave-one-out check of `gallery` on
the loaded team and keeps a strip under the banner until the gallery changes:
the score, the two most frequent misses (`app.leave_one_out_text`) and the
operatives that need more photos (`app.weak_text`). All misses and one `weak:`
line for each such operative go to the Terminal (see `gallery`). The banner shows the camera frame size and the measured frame rate of the window.

Zoom: `+` (or `=`) and `-` change a digital zoom around the mouse pointer, `0`
removes it (`live.ZoomView`). It does not add camera detail, because the crop
still comes from the full frame. While zoomed, the window shows the camera
view (`v` has no effect) and `space` searches only base centres inside the
shown rectangle (`pipeline.BasesSearch.area_px`).

`space` first runs the normal search, with only the base size of the operative
when `team.toml` has one. The size is then known, so this search keeps the
radius of that size ("Base detection", known sizes). When that finds no base, a
second search runs (`BasesSearch.relaxed`): the same fixed-radius fit, the
thresholds are `bases.relaxed_*`, and the empty-board reference is not used. It is a guess, so it never adds a crop by itself: the user clicks a
yellow circle. Without zoom it searches the full board and the true base is
less often in the shown circles. On the test photos, with a search area around
each base: 15 of 16 bases found, 14 in the first 5 (normal search: 10).

When a corner card is not seen, a red strip
names the missing cards (`app.corner_warning`, `app._CornerCheck`).

`--images`: takes the crop from saved photos. A photo with no base or more
than one base of the size is skipped with a message. So is a photo that the
gallery has already. Exit 0 when at least one
crop was added, 1 when none, 2 on a bad input or name. A name must not be
empty, start with `.`, or hold `/`, `\` or `:`. A `--base-mm` that differs
from `team.toml` is an error.

## `gallery`: check the gallery

```
uv run battlesight gallery --team NAME [NAME2] [--gallery-root data/teams] [--config FILE]
```

Lists each operative with its base size and photo count. Then it checks each
photo against the rest of the gallery (leave-one-out, size limit on, no
one-to-one assignment) and prints the score and the misses (`A looked like
B`). Then one `weak:` line for each operative with 2 wrong photos or more that
are 10 % of its photos or more (`app.weak_operatives`), worst first: the wrong
count, the operatives it looks like, and the advice to add photos of it.

## `calibrate`: lens calibration

```
uv run battlesight calibrate [--source S] [--camera-file data/camera.toml]
                             [--images PATH [PATH ...]]
                             [--min-views 10]
```

Do this one time for each camera mode (resolution, lens, zoom). Without a
calibration, the app uses a nominal lens: focal length from `focal_35mm` in
`data/camera.toml` (default 24 mm), no distortion. That is good for phones that correct the
distortion in the phone, but a calibration is better.

1. Print the target: `uv run tools/make_charuco.py`. This writes
   `data/print/charuco.pdf` (A4 landscape, ChArUco 10 x 7, 25 mm squares).
   Print at 100 % or fit to page. The print size does not change the lens
   values (focal length, centre, distortion): they are in pixels and do not
   depend on the square size. The scale must be the same in both directions.
2. Glue the print to something flat and stiff.
3. Run the command with the same camera as for play. It uses the saved
   camera file (`--source` for another source).
4. Hold the target in the camera view. Green dots show the found corners.
   Push `space` to keep the view. Keep 10 to 20 views: the target at the
   centre, at each edge and corner of the frame, some tilted 20 to 40 deg.
5. Push `c`. The command calibrates and writes `[lens]` to the camera file.
   The other values in the file stay.

With `--images`, the command reads the views from image files or folders.
All images must have the same size.

| Option | Default | Meaning |
|---|---|---|
| `--images` | none | Image files or folders. Use them instead of the camera. |
| `--source` | from the camera file | Camera for this run. |
| `--camera-file` | `data/camera.toml` | Camera file. The camera values come from it, and the command writes `[lens]` to it. |
| `--min-views` | `10` | Fewest views with the target. The command always needs at least 3. |

The command prints the reprojection error (`rms`). Under 0.5 px is good.

Exit codes: `0` calibration written, `1` too few views, `2` bad input or no
camera.

The app uses `[lens]` only for frames with the same aspect ratio as the
calibration images. It scales it to other sizes with that aspect ratio (4K
calibration, 1080p frames). For other frames it uses the nominal lens.

## `image-pipeline`: saved photos

```
uv run battlesight image-pipeline PATH [PATH ...] [--config FILE] [--camera-file FILE]
                                  [--reference FILE] [--team NAME [NAME2]] [--gallery-root DIR]
                                  [--out DIR] [--json FILE|-] [--show] [-q]
```

The command always runs all stages.

`PATH` is an image file or a folder. For a folder, the command reads all
images directly in it (not in subfolders), in name order. Accepted types:
`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`. The command applies
the EXIF orientation of phone photos. A folder does not give `<name>.empty.*`
files: these are empty-board references. To run the stages on a reference,
use its file name as `PATH`.

| Option | Default | Meaning |
|---|---|---|
| `--config` | `data/board.toml` | Board and tag config. |
| `--camera-file` | `data/camera.toml` | Camera file. Only the lens (`focal_35mm`, `[lens]`) is used. The photo tests do not read it: they always use the nominal 24 mm lens. |
| `--reference FILE` | none | Empty-board photo (terrain set up, no operatives) from the same camera and light. The `bases` stage then runs in reference mode for all images. An unreadable file, or one where `corners` or `rectify` fails, gives exit code 2. |
| `--team NAME [NAME2]` | all teams | One or two gallery teams for the `ident` stage. Without it the command uses every team folder in `--gallery-root` (a folder with a `team.toml` or operative folders). An unreadable gallery, or more than two names, gives exit code 2. |
| `--gallery-root` | `data/teams` | Folder with one folder per team. No team folder in it: the `ident` stage names nothing and the model does not load. Point it at an empty folder to skip the names. |
| `--out DIR` | none | Write `<name>.overlay.jpg`, one `<name>.<view>.jpg` per stage view, and `<name>.json` for each image to `DIR`. The command makes `DIR` if it does not exist. |
| `--json FILE` | none | Write one report for all images to `FILE`. Use `-` to write it to stdout. |
| `--show` | off | Show each overlay in a window. Push a key for the next image. Push `q` or `Esc` to stop. |
| `-q`, `--quiet` | off | Do not print a status line for each image. |

### Examples

Check all test photos:

```
uv run battlesight image-pipeline tests/images
```

Write overlays and JSON files to a folder:

```
uv run battlesight image-pipeline tests/images --out out/
```

`--out` writes `<name>.overlay.jpg` (camera frame with all overlays),
`<name>.rectified.jpg` (top-down board with the map grid, from the
`rectify` stage), `<name>.bases.jpg` (top-down board with the found bases),
and `<name>.json`.

Find the bases with an empty-board reference:

```
uv run battlesight image-pipeline game.jpg --reference empty.jpg --out out/
```

Send the report to another tool:

```
uv run battlesight image-pipeline shot.jpg --json - | jq '.images[0].stages.corners.data.board_corners_px'
```

Look at the overlays of one photo:

```
uv run battlesight image-pipeline shot.jpg --show
```

### Output

Each image gives one status line on stdout. With `--json -`, the status lines
go to stderr, so stdout holds only the JSON.

```
ok   tests/images/tomb-world-1.jpg  corners ok: 4/4 tags
FAIL tests/images/blank.png  corners FAIL: 0/4 tags, missing [0, 1, 2, 3]
```

Errors (file not found, file is not an image) go to stderr as `error: ...`.

### Exit codes

| Code | Meaning |
|---|---|
| `0` | All stages passed on all images. |
| `1` | One or more stages failed on one or more images. All images were read. |
| `2` | An input is missing or is not an image, or no image was found. Bad options also give `2` (argparse). |

The command does all images before it stops, also after a failure. One bad
image does not stop the others.

### Stages

The pipeline runs the stages in a fixed order. Each stage uses the results of
the stages before it. When a stage fails, the command skips the stages after
it on that image.

| Stage | Priority | Passes when |
|---|---|---|
| `corners` | 1 | All four corner tags are found. |
| `rectify` | 2 | The board plane fits the 16 tag corners: the largest error is not more than `rectify.max_reproj_mm`. |
| `bases` | 3 | Always, when it runs (no bases is a valid result). Fails only when the reference has another image size. |
| `ident` | 4 | Always, when it runs. Without a gallery (no team folder) or with an empty gallery it names nothing. Views: `ident` (top-down image with names). |

Each new priority adds a stage to this list. The live app runs all stages
on `space` and the stages up to `bases` on `b` (`Pipeline.analyze(...,
until=...)`).

### JSON report

The `--json` report:

```json
{
  "schema": 1,
  "config": "data/board.toml",
  "stages": ["corners", "rectify", "bases"],
  "reference": null,
  "teams": ["Raveners"],
  "ok": true,
  "errors": [],
  "images": [ /* one image report for each image */ ]
}
```

One image report (also the content of `<name>.json` from `--out`):

```json
{
  "schema": 1,
  "source": "tests/images/tomb-world-1.jpg",
  "size": [3072, 4096],
  "ok": true,
  "stages": {
    "corners": {
      "ok": true,
      "skipped": false,
      "message": "4/4 tags",
      "ms": 210.4,
      "data": {
        "tags": {
          "0": {"corner": "top-left", "corners_px": [[x, y], ...], "side_px": 158.0}
        },
        "missing": [],
        "board_corners_px": [[x, y], [x, y], [x, y], [x, y]]
      }
    }
  }
}
```

| Field | Meaning |
|---|---|
| `reference` | Path of the `--reference` file, or `null`. Only in the combined report. |
| `teams` | Gallery teams that the `ident` stage used, or `null` without a gallery. Only in the combined report. |
| `schema` | Report format version. It increases when a field changes or is removed. A new stage or a new field in `data` does not change it. |
| `size` | Image `[width, height]` in pixels, after the EXIF rotation. |
| `ok` | `true` when all stages in the report passed. |
| `stages.<name>.skipped` | `true` when an earlier stage failed. |
| `stages.<name>.ms` | Run time of the stage. Do not compare it in tests. |
| `stages.<name>.data` | Results of the stage. Each stage has its own fields. |

Fields of the `corners` stage:

| Field | Meaning |
|---|---|
| `tags` | Found tags by id: board corner name, the four tag corners in pixels (tag order TL, TR, BR, BL), mean edge length in pixels. |
| `missing` | Ids of the tags that were not found. |
| `board_corners_px` | Board corners in pixels, order TL, TR, BR, BL. `null` for a corner without a tag. |

Fields of the `rectify` stage:

| Field | Meaning |
|---|---|
| `lens` | Lens in use: `source` (`calibrated` or `nominal`), image `size`, `fx`, `fy`, `cx`, `cy` in pixels, `dist` (OpenCV k1 k2 p1 p2 k3), `rms_px` of the calibration. |
| `H_mm_to_px` | 3 x 3 homography from board mm on the play surface to undistorted image pixels. Apply the lens distortion after it to get frame pixels. |
| `reproj_rms_mm`, `reproj_max_mm` | Fit error of the 16 tag corners, in mm. A large value means a tag card is off its corner, a tag is printed at the wrong size, or the board size in the config is wrong. |
| `camera_mm` | Camera centre: x and y in board mm, height above the table in mm. |
| `thickness_mm`, `surface_scale` | Board thickness from the config, and the scale it gives about the camera nadir (1.0 for thickness 0). |
| `board_corners_px` | Play surface corners in frame pixels, order TL, TR, BR, BL. |
| `rectified` | Top-down image: `px_per_mm`, `margin_mm` (table around the board), `size` in pixels. |
| `edges` | The board edges found in the top-down image, by name (`top`, `right`, `bottom`, `left`). `offset_mm`: distance from the nominal edge at its midpoint, positive into the board. `angle_deg`: tilt. `n`: samples used, `rms_mm`: fit error. `null` when not found. |
| `corner_angles_deg` | Board corner angles TL, TR, BR, BL from the found edges. `null` when an edge is missing. |
| `grid_lines` | Only with `grid.printed = true`. Offsets in mm of the printed grid lines from their nominal places: `x` for the vertical lines (left to right), `y` for the horizontal lines (top to bottom). Positive = right or down. `null` for a line that is hidden too much (under 10 samples, or the samples do not agree). Dark edges of walls and models do not count as samples: the board must be the same on both sides of the line. |

The edge and grid offsets are checks. They do not make the stage fail. A
large edge offset means the board is not on the L lines of the cards (the
board slid, or the board size in the config is wrong).

Fields of the `bases` stage:

| Field | Meaning |
|---|---|
| `mode` | `reference` (an empty-board reference is set) or `single_frame`. |
| `px_per_mm` | Scale of the top-down image for the circle fit (`bases.px_per_mm`). |
| `diameters_mm` | The size classes of the search: the base sizes of the teams when they are known ("Base detection", known sizes), else the config values. |
| `changed_fraction` | Reference mode: part of the board that differs from the reference. A value near 1 means other light or a moved camera: take the reference again. `null` in single-frame mode. |
| `bases` | Found bases, best rim coverage first. Each: `center_mm` (board mm), `center_px` (frame pixels), `diameter_mm` (size class), `fit_diameter_mm` (free circle fit before the class was chosen), `coverage` (0..1, part of the rim that supports the fit), `rms_mm` (fit residual), `contrast` (median Lab L step from the rim to the board). |

Fields of the `ident` stage:

| Field | Meaning |
|---|---|
| `teams` | Gallery teams, or `null` without a gallery. |
| `model` | Embedder (`ident.model`). |
| `gallery_photos` | Number of gallery crops. |
| `masked` | Number of bases compared with a masked crop (only with a gallery that has photos). |
| `bases` | One entry per found base, in the order of the `bases` stage: `center_mm`, `diameter_mm`, `label` (`<team>/<operative>` or `null` = unknown), `score` (mean cosine similarity of the `top_k` best gallery photos), `margin` (score minus the best other allowed name; negative when the assignment took a second choice; `null` when there is no other), `sure`, `confirmed`, `candidates` (up to 5 `[label, score]`, best first, only operatives of this base size). |

## Operative identity

`src/battlesight/ident.py`, stage `ident`. Plan priority 4. Detection and
identity stay separate: the stage only reads the bases of the `bases` stage.

1. Crop: square around the base in the bases image (`bases.px_per_mm`). The
   gallery keeps the crop with 40 mm of board on each side
   (`STORE_PAD_MM`). The model input is the centre square of base diameter +
   2 x `crop_pad_mm`, at `input_px`, grey outside the inscribed circle.
2. Embedding: DINOv2 (`model`, Hugging Face name; the first run downloads it
   to `~/.cache/huggingface`). Class token and mean patch token, each L2
   normalised, joined. Runs on Apple MPS when available. About 80 to 90 ms
   for 10 bases on an M4 Pro. `model = "color-hist"` is a colour histogram
   without ML, for the tests.
3. Gallery: `data/teams/<team>/team.toml` (`[operatives."Name"]`,
   `base_mm`, optional `wounds` = maximum wounds for the live game state, not
   used by the identification), `data/teams/<team>/<operative>/*.png` crops. Each crop is
   embedded at `rotations` angles. `embeddings.npz` in the team folder caches
   the embeddings by file name and change time, with a key of the model and
   the crop values; a change of these values embeds all crops again. Git
   ignores the cache. `<operative>/bg/` holds the optional empty-board crop of
   each photo, from `enroll` or a name fix with a reference.
4. Score: per operative, the best rotation of each photo, then the mean of
   the `top_k` best photos. Only operatives whose `base_mm` is the size class
   of the base. One-to-one assignment (Hungarian, `scipy`) because each
   operative is on the board at most once. Score under `min_score`: unknown.
5. Fixes: `LabelSession` (no window calls, unit tested) holds the snapshot.
   A chosen name adds the crop to the gallery at once.
6. Mask (`ident.mask = true`, `mask_crop`): grey on all pixels that are not the
   operative, from the difference between a stored crop and its empty-board
   crop (`circles.foreground_mask`). Close gaps of 3 mm, cut out other found
   bases, always keep the base disc, keep only the part joined to the base,
   fill holes. No mask for a crop (`None`) when under `mask_min_base` of the
   base disc changed (the "empty" photo shows the operative) or when the kept
   part is over `mask_max_fraction` of the crop circle (light change, old
   photo). Gallery: each photo has raw and masked embeddings (`Gallery.view_emb`,
   `masked_emb`; both in the cache, NaN rows = no mask). Game: a base with a
   usable mask compares masked crops with `masked_emb`, other bases compare
   full crops with `view_emb`. No reference in the game, or an operative with
   gallery photos but no masked photo (`Gallery.can_mask` false): full crops
   for all bases. `leave_one_out` follows the same rule. Masked crops share
   more grey, so scores can be higher than with full crops: `min_score` and
   `min_margin` are not tuned for masks yet.

`[ident]` in `data/board.toml`:

| Key | Default | Meaning |
|---|---|---|
| `model` | `facebook/dinov2-small` | Hugging Face model, or `color-hist`. |
| `crop_pad_mm` | 25 | Board around the base in the model input, each side, at most 40. The model body leans out of the base by parallax (up to about 19 mm at a board corner). |
| `input_px` | 224 | Model input size. A multiple of 14 for DINOv2. |
| `rotations` | 8 | Angles per gallery crop. |
| `top_k` | 3 | Gallery photos per operative in the score. |
| `min_score` | 0.5 | Unknown under this score. |
| `min_margin` | 0.03 | Unsure under this lead over the second name. |
| `mask` | `false` (`true` in `data/board.toml`) | Use the mask of step 6. `false`: full crops only. A change embeds all crops again. |
| `mask_threshold` | 25.0 | Lab colour distance from the empty board that counts as foreground. |
| `mask_min_base` | 0.25 | No mask under this changed part of the base disc. |
| `mask_max_fraction` | 0.6 | No mask over this kept part of the crop circle. |

First check on the two test photos (no real gallery yet, each found base
enrolled as its own operative): the board around the base makes crops of
different operatives similar (cosine 0.5 to 0.87), and one model in both
photos scored 0.75. The done test (9 of 10 right) needs a gallery of 10 to
20 crops per operative from the real camera, and a labelled game photo
(`docs/testing.md`). Tune `crop_pad_mm`, `top_k`, `min_score` with it.

## Base detection

`src/battlesight/circles.py`, plan priority 3. Geometry only, no training.
The stage warps the frame to a top-down image at `bases.px_per_mm` (default
4 px per mm, the camera gives about 3.5 to 4) with a 5 mm border. Then:

1. With a reference: Lab distance to the reference image (warped the same
   way from its own tags), blur, threshold `bases.diff_threshold`. Centres
   must lie near changed pixels, and `bases.min_foreground` of the base disc
   must be changed.
2. Proposals: each strong edge pixel with a dark band inside (at 0.5 and
   1.0 mm) and lighter board outside (at 1.25 mm) votes for the centre one
   radius inward along its gradient, for each diameter. The peaks of the
   vote maps are the proposals (`max_proposals` per diameter).
3. Verification: 120 rays from the proposal. On each ray the rim edge is the
   strongest dark-to-light step within 3.5 mm of the radius that passes the
   rim test: `outer - rim > bases.min_contrast` and
   `rim < bases.rim_ratio * outer`. A robust free circle fit gives the centre
   and the fitted diameter. Three rounds, each with rays from the new centre.
4. Class: the nearest diameter in `bases.diameters_mm`, when it is within
   `bases.diameter_tol_mm`. A last fit with the class radius gives the
   centre. `coverage` is the part of the 120 rays with an edge within 1 mm
   of that circle. It must be at least `bases.min_coverage`.
5. Overlapping circles: the best coverage wins. Bases can touch, so two
   circles overlap when their centres are closer than 0.8 times the sum of
   the radii.

The diameter is the bottom of the base, where it touches the board. The
bottom edge lies on the play surface, so the rectify homography maps it
without parallax. The top of the base is smaller and higher; the rim test
uses only the dark bevel next to the bottom edge. On the far side of a base
at a board corner (about 35 deg view angle) the top edge can hide the
bottom edge by about 0.5 mm.

Known sizes. The stage knows the base sizes on the board when it has a
roster: `Pipeline(roster=...)` or `set_roster`, else the sizes of the gallery
operatives (`Gallery.base_sizes`, from `team.toml`), else the one size of an
`enroll` search. The live app sets the roster from the team files at the start
(`ident.roster_sizes`), so the `b` key has it before the gallery loads. An
operative with no size in `team.toml` gives no roster. With known sizes
(`pipeline.detect_bases`):

- Only these sizes are searched. A base of another size is not found. On the
  test photos this removes 8 of 10 false bases (no team there has 25, 28 or
  50 mm bases).
- Steps 3 and 4 keep the radius of each size from the start
  (`circles._verify_fixed`). A free fit goes to the model or its shadow when
  a part of the rim is hidden: 7 of the 9 missed bases of the test photos have
  a fixed-radius fit with a coverage of 0.41 to 0.52, less than 1.1 mm off.
- With a reference too, the coverage limit is `bases.reference_min_coverage`
  (0.4). The changed area then stops the false circles. Without a reference
  this value gives 20 to 30 false bases on a photo, so `min_coverage` stays.

Without known sizes the fit is free, as in steps 3 and 4: sizes that are 3 mm
apart cannot be told apart with a fixed radius.

Single-frame mode (no reference) also finds round things with a dark rim:
tokens with a dark ring, round barricades, holes in a wall grating. Bases
with less than `min_coverage` of the rim visible (a model or a wall over
it) are not reported. Reference mode removes the static ones. Plan section
9 has the results on the test photos. `tools/bench_bases.py` (below)
measures them.

`[bases]` in `data/board.toml`:

| Key | Default | Meaning |
|---|---|---|
| `diameters_mm` | `[25, 28, 32, 40, 50]` | Size classes, base bottom. |
| `px_per_mm` | `4.0` | Scale of the top-down image for the fit. |
| `diameter_tol_mm` | `3.0` | Largest distance of the fitted diameter from a class. |
| `min_coverage` | `0.5` | Smallest visible part of the rim. |
| `min_contrast` | `12.0` | Smallest Lab L step (0..255) from the rim band to the board. |
| `rim_ratio` | `0.75` | Rim band L at most this part of the board L outside. |
| `diff_threshold` | `25.0` | Reference mode: Lab distance that counts as changed. |
| `min_foreground` | `0.25` | Reference mode: changed part of the base disc. |
| `reference_min_coverage` | `0.4` | `min_coverage` in reference mode with known base sizes. |
| `relaxed_min_coverage` | `0.3` | `enroll` second search: `min_coverage`. |
| `relaxed_min_contrast` | `6.0` | `enroll` second search: `min_contrast`. |
| `relaxed_rim_ratio` | `0.9` | `enroll` second search: `rim_ratio`. |
| `relaxed_max_found` | `5` | `enroll` second search: most circles shown, best coverage first. |

## Map grid

The grid that `rectify` and the overlays draw depends on the map. Set it in
the `[grid]` section of `data/board.toml`. All values are in board mm.

| Map | Grid | Config |
|---|---|---|
| Tomb World | 7 x 6 printed sections inside a 12 mm margin. Each section is 97 mm. An inch grid does not agree with the print. | `margin_mm = 12.0`, `sections = [7, 6]`, `printed = true` |
| Volkus | 1 in squares from the board edge. | `margin_mm = 0.0`, `pitch_mm = 25.4`, no `sections` |

| Key | Default | Meaning |
|---|---|---|
| `margin_mm` | `0.0` | Distance from each board edge to the first grid line. |
| `sections` | none | `[nx, ny]`: split the space inside the margin into nx by ny equal sections. The lines include both margin lines. |
| `pitch_mm` | `25.4` | Line spacing when `sections` is not set. The lines start at the margin. |
| `printed` | `false` | The grid lines are printed on the board. The `rectify` stage then finds them in the top-down image and reports `grid_lines`: a check of the rectification that does not use the tags. |

The Tomb World values come from the test photos: the thick section lines are
12 mm from each edge of a 703 x 606 mm board, (703 - 24) / 7 = (606 - 24) /
6 = 97 mm apart. For a new map, find the lines in `out/<name>.rectified.jpg`
and set the margin and sections or pitch.

## Terrain maps

`terrain.py` reads map files: one layout of one terrain type per file, in
`data/maps/<terrain folder>/<layout>.toml` (committed). An agent writes them
from a photo of the layout card with the project skill
`.claude/skills/record-map/` (reading rules there). No detection code.

The live app has no key and no option for maps. `MapPicker` (`ui/parts.py`)
at the top of the panel of each layout sends `map.terrain_type` and
`map.layout` (`name = None` = no layout); `view.terrain` ("Layout", on at
the start) shows or hides walls, pillars, openings, discs and light terrain; `view.zones`
("Drop zones", off at the start) shows or hides the zones; `view.keys`
("Terrain keys", off at the start) shows or hides the wall keys (above a
horizontal wall, right of a vertical wall), `T` on each pad and `C1`.. on
each light terrain piece. `view.minimap` ("Minimap", off at the start, no
key) shows a drawn board in place of the camera image
(`LiveSession._render_minimap`): the layout (also with "Layout" off), the
grid when "Grid" is on, a full disc for each base of the last snapshot (team
colour, grey without a name), names, health bars and the selection with its
measurements. It has the size and scale of the top-down view, so clicks and
`position_px` work the same. It covers the live video and the snapshot:
`state["mode"]` does not change, `state["minimap"]` is true. It shows also
while the camera is down. It goes off when no layout is left. All four are
enabled only with a
layout. State: `terrain_types`, `terrain_type`, `map_layouts`, `map_layout`,
`terrain`, `zones`, `keys`. A bad file gives a notice, never a stop. `overlay.draw_terrain`
draws the shapes in the live view, both views, and the snapshot.

### File format

```toml
schema = 1
name = "Layout 6"            # shown in the "Layout" list
terrain = "Tomb World"       # shown in the "Terrain type" list
source = "layout card 6, photo"

pillars = [[0, 1], [1, 1], [3, 1], [4, 1]]   # grid points
wall_ends = [[6, 3]]         # half pillar, past the end of its one wall

[[wall]]
key = "A1"                   # wall label on the card: A1..A4 (2 grid), B1..B4 (1 grid)
from = [3, 1]                # pillar centres
to = [5, 1]
opening = true               # hexagon marker next to the wall; A1, B2: breach point, others: hatchway
opening_at = [4.5, 1]        # needed when a pillar is at the wall middle

[[objective]]
at = [5, 2]                  # or at_mm = [x, y] in board mm

[[teleporter]]
at = [2, 2.5]

[[zone]]
name = "grey"                # "grey" | "orange" (fill colour)
from = [0, 0]                # top-left grid point
to = [7, 1]                  # bottom-right; a side on an outer grid line goes to the board edge

[[light]]
key = "C2"                   # light terrain label on the card, a key of [terrain] light_mm
at = [5, 4]                  # centre; or at_mm = [x, y]
long = "y"                   # board axis of the long side: "x" (default) or "y"
```

Light terrain (`C1`..`C5`) is the footprint of the card symbol (rectangle,
C1 an octagon), not the real outline of the piece. It is drawn only: it is
not in `walls.obstacles`, so it has no effect on measurement around walls or
on movement.

A grid point `[col, row]` is an intersection of the `[grid]` lines (index
from 0, steps of 0.5; Tomb World col 0..7, row 0..6). `at_mm` is board mm.

`load_map` raises `MapError` when: the schema is not 1, `name` or `terrain`
is missing, a key is unknown, a point is not on the half grid or is outside
the grid or board, a wall is not horizontal or vertical, a wall `key` is not
`A<n>` or `B<n>`, an A wall is not 2 grid or a B wall not 1 grid long, an A
wall (except `no_middle_pillar`) has no middle pillar or another wall has
one, `opening` is not true or false, `opening_at` is given with no opening, two walls are the same, a wall end has no pillar or wall end, a
wall end touches not exactly 1 wall, two pillars share a point, a pillar
inside a wall is not at its middle, an opening of a wall with a middle pillar
has no `opening_at`, `opening_at` is not the centre between two pillars, a
zone has `from` not top-left of `to`, a light `key` is not in
`[terrain] light_mm`, `long` is not `"x"` or `"y"`, a light footprint goes
more than 1 mm over the board edge.

### Shapes and `[terrain]` sizes

`shapes(map, cfg)` gives the footprints in board mm, with the sizes of
`[terrain]` in `data/board.toml`:

| Key | Default | Shape |
|---|---|---|
| `wall_mm` | `4.0` | Wall: rectangle pillar centre to pillar centre, this thick. |
| `pillar_mm` | `13.0` | Pillar: square on the grid point. |
| `wall_end_mm` | `6.5` | Wall end: `pillar_mm` across the wall, this along it, from the point away from the wall (it adds to the wall length). |
| `hatchway_width_mm`, `hatchway_depth_mm` | `22.0`, `10.0` | Hatchway: at the opening centre, width along the wall. |
| `breach_long_mm`, `breach_short_mm` | `35.0`, `28.0` | Breach point on a 2-grid (A1) or 1-grid (B2) wall, as thick as the wall. |
| `objective_mm` | `40.0` | Objective marker diameter. |
| `breach_walls` | `["A1", "B2"]` | Wall keys whose opening is a breach point; the opening of another wall is a hatchway. |
| `no_middle_pillar` | `["A2"]` | A walls with no pillar at the middle. |
| `objective_range_mm` | `25.4` | Dashed circle this far from the marker edge (1 in), shape `objective_range`. |
| `teleporter_mm` | `60.0` | Teleporter pad diameter. |
| `light_mm` | `{ C1 = [81, 63, 16], C2 = [62, 46], C3 = [48, 48], C4 = [45, 45], C5 = [40, 40] }` | Light terrain: `[long, short]` or `[long, short, corner cut]` in mm. A corner cut gives an octagon. Shape `light` (half see-through dark fill), key `light_key`. |

### `map` command

```bash
uv run battlesight map data/maps/tomb-world/layout-6.toml --out out/maps/
```

Checks each file and prints its counts. With `--out`, writes
`<file>.png`: the map on an empty top-down board with the grid numbers.
Exit code 0 all files good, 1 a file has an error, 2 a file is missing.

## Automated tests

[testing.md](testing.md) tells about the unit tests and the photo tests. The
photo tests use the `<name>.json` files from `--out` as expected results.
The bases benchmark (`tools/bench_bases.py`, below) gives metrics and a
baseline for the bases stage. The identification benchmark
(`tools/bench_ident.py`, below) gives them for the `ident` stage.

In CI or a shell script, use the exit code of the `image-pipeline` command:

```
uv run battlesight image-pipeline tests/images -q --json report.json || echo "pipeline failed"
```

## Add a stage

Add the stage in `src/battlesight/pipeline.py`, after the existing stages:

```python
def _draw_tokens(frame, result, cfg):
    ...  # draw in place on the BGR camera frame

def _views_tokens(frame, result, cfg):
    return {"tokens": ...}  # optional extra images, written as <name>.tokens.jpg

@stage("tokens", draw=_draw_tokens, views=_views_tokens)
def _tokens(ctx: Context) -> StageResult:
    view = ctx.scratch["view"]              # BoardView from the rectify stage
    top_down = ctx.scratch["rectified"]     # top-down image, no overlays
    bases = ctx.scratch["bases"]            # list of circles.Circle from the bases stage
    ...
    return StageResult(ok=True, message="...", data={...})
```

Rules:

- `data` must be JSON-safe: numbers, strings, lists, dicts, `None`. Put NumPy
  arrays and other objects in `ctx.scratch` for later stages and for drawing.
- Put expensive objects (detectors, models) in `ctx.cache`. The cache stays
  for all images of one command.
- The stage must not open a window or read a camera.
- `views` is optional. It returns extra named images. `--out` writes each as
  `<name>.<view>.jpg`, `--show` shows each in its own window.

The `image-pipeline` command, the live app, and `test_photo_passes_all_stages`
use the new stage with no other change. Add the stage to the "Stages" table above.
Then update the photo test expected files (see [testing.md](testing.md),
"New stages").

## `tools/bench_bases.py`: bases benchmark

Plan: `specs/bases-benchmark.md`. The tool measures the `bases` stage on
photos with ground truth (`<name>.bases.json`). It gives numbers, not only
pass or fail. Use it to tune `[bases]` and to stop changes that make the
result worse.

```
uv run tools/bench_bases.py [PATH ...] [--config data/board.toml]
                            [--mode auto|single|reference]
                            [--set KEY=VALUE ...] [--sweep KEY=V1,V2,... ...]
                            [--out out/bench] [--json FILE|-]
                            [--baseline FILE] [--write-baseline FILE]
                            [--no-images] [-q]
```

| Option | Default | Meaning |
|---|---|---|
| `PATH` | `tests/images` | Photos or folders. Only photos with a `<name>.bases.json` count. `<name>.empty.*` is a reference, not a photo. |
| `--config` | `data/board.toml` | Board config. |
| `--mode` | `auto` | `auto`: reference mode when `<name>.empty.*` exists. `single`: never use the reference. `reference`: only photos with a reference. |
| `--set` | none | Override a `[bases]` key for this run: `--set min_coverage=0.4 --set diameters_mm=32,40`. Unknown key or bad value: exit 2. A `diameters_mm` value (also in `--sweep`) turns off the roster of the photos: the search is then free over these sizes. |
| `--sweep` | none | Run all combinations. Values split by `,`. For `diameters_mm` split the alternatives by `;`: `--sweep "diameters_mm=32,40;25,28,32,40,50"`. Prints one total row per combination, best `f1` first, then fewer `false`. The table, `bench.json` and the images are for the best one. A `px_per_mm` sweep makes a new bases image for each value (slow). |
| `--out` | `out/bench` | Folder for `<name>.bench.jpg` and `bench.json`. |
| `--json` | none | Also write `bench.json` to a file, or `-` for stdout (the table then goes to stderr). |
| `--baseline` | none | Compare with a baseline file. Exit 1 when a rule is worse (see [testing.md](testing.md), "Bases benchmark"). |
| `--write-baseline` | none | Write the metrics as a new baseline. Keeps the `holdout` list of an existing file. |
| `--no-images` | off | No debug images (faster sweeps). |
| `-q` | off | No progress lines on stderr. |

The tool uses the nominal lens, as the photo tests do. It does not read
`data/camera.toml`. It runs `corners` and `rectify` once per photo and per
reference, and then only the base detection (`pipeline.detect_bases`) for
each combination.

Exit codes: 0 done, 1 worse than `--baseline`, 2 bad input (no photo with
ground truth, unreadable file, a photo or reference that fails `corners` or
`rectify`, bad `--set` or `--sweep`, bad ground truth file).

Text output: one row per photo, `TOTAL`, `target` (the priority 3 done
test), false bases by kind, recall by rim. Then one line for each miss
(`MISS`), wrong size (`SIZE`) and false base (`FALSE`), with the board mm,
so you can find it in the debug image.

`<name>.bench.jpg`: the top-down image at `rectify.px_per_mm` with green
circles (match, right size), orange (match, wrong size, `found/true`, line
to the true centre), red (false base, kind and coverage), yellow dashed
(missed base), grey dotted (`not_bases` entry). The banner has the photo
result.

`bench.json`:

```json
{
  "schema": 1,
  "config": {"diameters_mm": [25.0, 28.0, 32.0, 40.0, 50.0], "min_coverage": 0.5, "...": "all [bases] values used"},
  "overrides": {"min_coverage": 0.4},
  "git": "f56156b+dirty",
  "photos": [
    {"name": "tomb-world-1", "mode": "single_frame", "changed_fraction": null,
     "metrics": {"truth": 8, "found": 8, "matched": 6, "...": "..."},
     "by_rim": {"clear": {"...": "..."}, "hidden": {"...": "..."}},
     "matches": [{"truth": {"...": "..."}, "found": {"...": "..."}, "error_mm": 0.04, "size_ok": true}],
     "missed": [{"...": "truth entry"}],
     "false": [{"found": {"...": "..."}, "kind": "token"}]}
  ],
  "total": {"metrics": {"...": "..."}, "by_rim": {"...": "..."}, "by_mode": {"...": "..."}, "false_by_kind": {"other": 6}},
  "sweep": [{"overrides": {"...": "..."}, "total": {"...": "..."}, "holdout": null}]
}
```

`git` is `git rev-parse --short HEAD`, with `+dirty` when the work tree has
changes, or `null`. `sweep` is there only with `--sweep`. `holdout` (a list
of photo names) is there when the baseline has one. The metric fields are
in [testing.md](testing.md), "Bases benchmark".

Examples:

```
uv run tools/bench_bases.py                                            # all test photos
uv run tools/bench_bases.py --set diameters_mm=32,40                   # only the sizes of the teams on the photos
uv run tools/bench_bases.py --sweep min_coverage=0.4,0.45,0.5 --no-images
uv run tools/bench_bases.py --baseline tests/images/bases_baseline.json
```

## `tools/bench_ident.py`: identification benchmark

Plan: `specs/ident-benchmark.md`. The tool measures the operative names on photos with `<name>.ident.json`. It
gives numbers, not only the pass or fail of the done test. Use it to tune
`[ident]` and to stop changes that make the names worse.

```
uv run tools/bench_ident.py [PATH ...] [--config data/board.toml]
                            [--gallery-root data/teams] [--json FILE|-]
                            [--baseline FILE] [--write-baseline FILE] [-q]
```

| Option | Default | Meaning |
|---|---|---|
| `PATH` | `tests/images` | Photos or folders. Only photos with a `<name>.ident.json` count. `<name>.empty.*` is a reference, not a photo. |
| `--config` | `data/board.toml` | Board config. |
| `--gallery-root` | `data/teams` | Folder with one folder per team. The teams come from each `<name>.ident.json`. |
| `--json` | none | Also write the report to a file, or `-` for stdout (the table then goes to stderr). |
| `--baseline` | none | Compare with a baseline file. Exit 1 when a rule is worse (see [testing.md](testing.md), "Identification benchmark"). |
| `--write-baseline` | none | Write the metrics as a new baseline. |
| `-q` | off | No progress lines on stderr. |

The tool runs all stages with the nominal lens and the empty-board reference
when there is one, as the done test does. It does not read
`data/camera.toml`. The code is in `src/battlesight/ident_bench.py`.

Copies of a test photo. A name fix in the live app on a test photo
(`--image-source`) puts crops of that photo in the gallery. With such a crop
the benchmark only finds the same image again (score 1.00). For each photo the
tool finds the gallery photos that show the same frame as a found base
(`ident_bench.copies_of`: half-size crops, a small move from the phase
correlation, under 0.3 % of the centre pixels more than 32 levels off) and
names the bases again without them (`Gallery.hidden`). The files stay in the
gallery. The table has one line with the count, the metrics have `hidden`.

Exit codes: 0 done, 1 worse than `--baseline`, 2 bad input (no photo with
names, unreadable file, a team with no gallery or no gallery photos, a failed
stage, a baseline of another gallery).

Text output: one row per photo, `TOTAL`, and the target. Then one line for
each wrong name (`WRONG`), unknown name (`UNKNOWN`), wrong base size (`SIZE`)
and missed base (`MISS`).

The report:

```json
{
  "schema": 1,
  "config": {"model": "facebook/dinov2-small", "top_k": 3, "...": "all [ident] values used"},
  "gallery": {"teams": ["Deathwatch", "Raveners"], "files": 554, "fingerprint": "9cbf949945da25ac"},
  "git": "9035073+dirty",
  "photos": [
    {"name": "tomb-world-1", "mode": "single_frame",
     "metrics": {"truth": 7, "right": 4, "hidden": 0, "...": "..."},
     "wrong": [{"true": "Raveners/Felltalon", "got": "Raveners/Prime", "sure": false}],
     "unknown": [],
     "wrong_size": [{"true": "Deathwatch/Sergeant", "found_mm": 28.0, "true_mm": 32.0}],
     "missed": ["Raveners/Prime"]}
  ],
  "total": {"metrics": {"...": "..."}}
}
```

A baseline has the same fields, but no `wrong`, `unknown`, `wrong_size` and
`missed` lists. The metric fields are in [testing.md](testing.md),
"Identification benchmark".

Examples:

```
uv run tools/bench_ident.py                                              # all test photos with names
uv run tools/bench_ident.py --baseline tests/images/ident_baseline.json
uv run tools/bench_ident.py --write-baseline tests/images/ident_baseline.json
```

## `tools/annotate_bases.py`: click the ground truth

```
uv run tools/annotate_bases.py tests/images/<name>.jpg [--config data/board.toml] [--prefill] [--team NAME ...]
                               [--display-width 1600] [--display-height 950]
```

The tool runs `corners` and `rectify` (nominal lens), makes the bases image
(`bases.px_per_mm`), and shows it in a window. It loads
`<name>.bases.json` when it exists and writes it (schema 2) on `s`. The
ground truth then does not come from the detector.

| Key | Action |
|---|---|
| left click | Add a point on the bottom rim of a base. 5 or more around the visible part is best. |
| `Enter` | 3 or more points: fit a free circle and snap the diameter to the nearest class. `source: "clicked"`, `tol_mm` 2, the points go in `points_mm`. The rim is `clear` when the points span at least 180 deg. Fewer than 3 points: an estimate at the click, `source: "estimate"`, `tol_mm` 8, rim `hidden`. |
| `1` to `9` | With points: fit with size class 1 to 9 of `bases.diameters_mm`. Without points: set the size of the base under the mouse. |
| `0` | Size unknown (`null`). |
| `Backspace` | Drop the points. |
| `h` | Rim `clear` <-> `hidden` (base under the mouse, else the last one). |
| `t` | Note for the entry under the mouse. Type it in the terminal. |
| `x` | `not_bases` entry. With 3 or more points: of their fit. Else: of the base or the proposal under the mouse (the base moves, the proposal goes away). Type its kind in the terminal. |
| `n` | With `--team`: assign an operative to the base under the mouse. The terminal lists the operatives of that base size as 1 to 9. Push the number in the window. `0` clears the name, `n` shows the next page, `Esc` cancels. No other key works while the list is open. An operative already on another base is marked, but you can select it. |
| `a` | With `--prefill`: accept the detector proposal (cyan dashed) under the mouse, `source: "detector"`. The rim is `clear` when its coverage is 0.5 or more. |
| `p` | With `--prefill`: show all proposals. By default a proposal on a saved entry (within its `tol_mm`, at least 3 mm, or inside a `not_bases` circle) is hidden. |
| `d` | Delete the entry or proposal under the mouse. |
| `u` | Undo. |
| `+` / `-`, `z` / `Z`, mouse wheel | Zoom in / out at the mouse: 1x, 2x, 4x, 8x of the screen fit. |
| arrows, right-drag | Pan. |
| `s` | Save. |
| `q`, `Esc` | Quit. Asks in the terminal to save when there are changes. |

The banner shows the board mm under the mouse, the counts, the points, and
the free fit diameter, and the shown / all proposals. Green circles are clear bases, yellow hidden bases,
dashed circles estimates, cyan dashed circles detector proposals (not
saved), grey dotted circles `not_bases`, magenta the fit of the current
points. A label shows the size and the first letter of the source (`c`
clicked, `d` detector, `e` estimate).

With `--team` (teams in `data/teams/`) the tool loads `<name>.ident.json` when it
exists and writes it on `s` next to `<name>.bases.json`: the input of
the identification benchmark. Only named bases go in the file, each with the centre of its
base. The names are not in `<name>.bases.json`. On load, a name goes to the nearest
free base within `tol_mm`. A name with no base gives a warning and stays in the
file. A name of a team outside `--team` is an error (exit 2). The model is not
loaded without `--prefill`. The operative list comes from `team.toml` and the operative folders.

With `--prefill` and `--team` the tool also runs the identification of the app
(`Annotator.suggest`, the model loads at start) on the saved bases and the open
proposals. Each base with no name starts with the best guess, and `a` gives an
accepted proposal its guess. A name from the `.ident.json` stays, and a guess that
another base has is not set. `s` saves the guesses as names, so check each one:
an unchecked guess makes the done test compare the app with itself. A cyan `name?`
shows the guess on a proposal or an unnamed base. In the `n` list the guess is
key 1, then the other operatives by score. A new base or a new size runs the
suggestions again on the next `n`.

On `s` the tool also writes `<name>.expected.json` when it is missing, with the
stages up to `bases` and the empty-board reference when there is one (the same run
as `tests/test_photos.py`). It never replaces this file: the photo tests compare
later runs with it. To replace it, use `image-pipeline` and copy the report
(`docs/testing.md`, "Add a photo").

The edit logic is the `Annotator` class (no window calls). `tests/test_bench.py` drives it.

## `tools/list_cameras.py`: find and save the camera

```
uv run tools/list_cameras.py [--max 6] [--width 3840] [--height 2160]
uv run tools/list_cameras.py --use INDEX [--width 3840] [--height 2160] [--fps N]
uv run tools/list_cameras.py --url URL
```

| Option | Default | Meaning |
|---|---|---|
| (none) | | List the device indices `0` to `--max - 1`: default size, size after the request, frame rate. Saves nothing. |
| `--use` | none | Check this device index at `--width` x `--height` and save it. When the camera gives another size, the script saves the size it gives. |
| `--url` | none | Check this stream (30 frames, measured frame rate) and save the URL. Width and height are not saved for a stream. |
| `--width`, `--height` | `3840`, `2160` | Frame size to request from a device. |
| `--fps` | none | Frame rate to save with `--use`. |
| `--no-save` | off | Only check. Do not write the camera file. |
| `--camera-file` | `data/camera.toml` | Camera settings file to write. |

Exit code 0 when the camera works, 1 when it does not open or gives no frame.
The script writes only `[camera]`. `focal_35mm` and `[lens]` stay. When the
source changes and the file has a lens calibration, the script tells you to
calibrate again if the calibration is for another camera.

## Camera file

`data/camera.toml` holds everything about the camera. It is local to the
machine (it can hold the phone IP address), so git ignores it. Without the
file, the app uses the built-in defaults: camera 0, 1920 x 1080, nominal
24 mm lens.

```toml
[camera]                     # tools/list_cameras.py writes it
source = 1                   # device index, or a URL string such as "http://192.168.1.20:8080/video"
width = 3840                 # devices only
height = 2160                # devices only
fps = 30.0                   # optional
focal_35mm = 24.0            # nominal lens: 35 mm equivalent focal length

[lens]                       # battlesight calibrate writes it
image_size = [3840, 2160]    # frame size of the calibration views
camera_matrix = [[3012.4, 0.0, 1921.7], [0.0, 3010.9, 1078.2], [0.0, 0.0, 1.0]]
dist_coeffs = [0.0412, -0.1033, 0.0002, -0.0004, 0.0581]   # OpenCV k1 k2 p1 p2 k3
rms_px = 0.31                # reprojection error
views = 15
date = "2026-11-03 14:20:11"
```

`CameraSettings` in `src/battlesight/camera.py` reads and writes it. The
live commands and `image-pipeline` read the lens from it (`--camera-file`). The board
config (`data/board.toml`) must not have a `[camera]` table: the app stops
with an error that tells you to move it.

## Add a command

Add a `cmd_<name>(args)` function and a subparser in `main()` in
`src/battlesight/app.py`. Return the exit code from the function. Add the
command to "All commands" at the top and a section for it to this document.
Follow the option rule: no option for a thing that a key can change.
